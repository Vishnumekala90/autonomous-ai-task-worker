"""Reset and seed the simulated company environment: python scripts/seed_database.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.database.database import Database  # noqa: E402
from app.database.seed import SEED_INVOICES, seed_database  # noqa: E402


def main() -> None:
    settings = get_settings()
    seed_database(Database(settings.database_path), settings.invoices_dir, reset=True)
    print(f"Seeded {len(SEED_INVOICES)} invoices into {settings.database_path}")
    print(f"Invoice documents written to {settings.invoices_dir}")


if __name__ == "__main__":
    main()
