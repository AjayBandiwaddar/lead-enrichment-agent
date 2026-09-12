from dotenv import load_dotenv

load_dotenv()

from app.extraction import build_extraction_context


def main():
    print("Extraction context module loaded successfully.")
    print("Fields supported:")

    for field in (
        "overview",
        "icp",
        "contacts",
        "leadership",
    ):
        print(f"  - {field}")


if __name__ == "__main__":
    main()