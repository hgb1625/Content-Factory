"""
Test SQLite Schema Migration and ORM Compatibility.

Verifies:
1. Creating old Video schema without provider, canonical_source_id, media_hash.
2. Inserting rows into old schema.
3. Running migrate_schema(target_engine).
4. Verifying new columns exist and old rows remain intact with defaults.
5. Running migrate_schema a second time (idempotency check).
6. Full ORM operations: insert, read, update, duplicate lookup on new columns.
"""
import tempfile
import unittest
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.database import Base, migrate_schema
from app.models import Video, Product


class TestDBSchemaMigration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_migration_")
        self.db_path = Path(self.temp_dir) / "test_old_schema.db"
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        self.Session = sessionmaker(bind=self.engine)

    def tearDown(self):
        self.engine.dispose()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_old_schema_migration_and_idempotency(self):
        # 1. Create old table schema without provider, canonical_source_id, media_hash
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id VARCHAR(32) NOT NULL UNIQUE,
                    product_id VARCHAR(32),
                    douyin_url VARCHAR(1024) NOT NULL UNIQUE,
                    views VARCHAR(64),
                    thumbnail TEXT,
                    local_file TEXT,
                    downloaded BOOLEAN DEFAULT 0,
                    approved BOOLEAN DEFAULT 0,
                    used BOOLEAN DEFAULT 0,
                    status VARCHAR(64) DEFAULT 'FOUND',
                    created_at DATETIME,
                    notes TEXT
                )
            """))
            # Insert existing old row
            conn.execute(text("""
                INSERT INTO videos (video_id, product_id, douyin_url, status)
                VALUES ('V0001', 'P0001', 'https://www.douyin.com/video/7111111111111111111', 'FOUND')
            """))
            conn.commit()

            # Verify columns before migration
            res_before = conn.execute(text("PRAGMA table_info(videos)")).fetchall()
            cols_before = {r[1] for r in res_before}
            self.assertNotIn("provider", cols_before)
            self.assertNotIn("canonical_source_id", cols_before)
            self.assertNotIn("media_hash", cols_before)

        # 2. Run non-destructive migration
        migrate_schema(target_engine=self.engine)

        # 3. Verify new columns exist
        with self.engine.connect() as conn:
            res_after = conn.execute(text("PRAGMA table_info(videos)")).fetchall()
            cols_after = {r[1] for r in res_after}
            self.assertIn("provider", cols_after)
            self.assertIn("canonical_source_id", cols_after)
            self.assertIn("media_hash", cols_after)

            # 4. Verify existing row is preserved and has default provider
            row = conn.execute(text("SELECT video_id, douyin_url, provider, canonical_source_id, media_hash FROM videos WHERE video_id = 'V0001'")).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], "V0001")
            self.assertEqual(row[1], "https://www.douyin.com/video/7111111111111111111")
            self.assertEqual(row[2], "douyin")  # default value
            self.assertIsNone(row[3])
            self.assertIsNone(row[4])

        # 5. Run migration a second time (idempotent, must not fail)
        migrate_schema(target_engine=self.engine)

        with self.engine.connect() as conn:
            res_again = conn.execute(text("PRAGMA table_info(videos)")).fetchall()
            cols_again = {r[1] for r in res_again}
            self.assertIn("provider", cols_again)
            self.assertIn("canonical_source_id", cols_again)
            self.assertIn("media_hash", cols_again)

    def test_02_orm_compatibility_and_crud(self):
        # Create full schema via migration on empty db
        Base.metadata.create_all(bind=self.engine)
        migrate_schema(target_engine=self.engine)

        session = self.Session()
        try:
            # 1. Insert new Video with provider, canonical_source_id, media_hash
            v = Video(
                video_id="V0002",
                product_id="P0002",
                douyin_url="https://www.douyin.com/video/7222222222222222222",
                provider="pexels",
                canonical_source_id="pexels:987654",
                media_hash="sha256_mock_hash_002",
                status="DOWNLOADED"
            )
            session.add(v)
            session.commit()

            # 2. Read back and verify fields
            v_read = session.query(Video).filter(Video.video_id == "V0002").first()
            self.assertIsNotNone(v_read)
            self.assertEqual(v_read.provider, "pexels")
            self.assertEqual(v_read.canonical_source_id, "pexels:987654")
            self.assertEqual(v_read.media_hash, "sha256_mock_hash_002")

            # 3. Update fields
            v_read.media_hash = "sha256_updated_hash_002"
            session.commit()

            v_updated = session.query(Video).filter(Video.video_id == "V0002").first()
            self.assertEqual(v_updated.media_hash, "sha256_updated_hash_002")

            # 4. Duplicate lookup on canonical_source_id and media_hash
            dup_by_source = session.query(Video).filter(Video.canonical_source_id == "pexels:987654").first()
            self.assertIsNotNone(dup_by_source)
            self.assertEqual(dup_by_source.video_id, "V0002")

            dup_by_hash = session.query(Video).filter(Video.media_hash == "sha256_updated_hash_002").first()
            self.assertIsNotNone(dup_by_hash)
            self.assertEqual(dup_by_hash.video_id, "V0002")
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
