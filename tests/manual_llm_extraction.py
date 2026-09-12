from dotenv import load_dotenv

load_dotenv()

from app.extraction import StructuredExtractor


def main() -> None:
    evidence = """
[page-1] Postman company overview
URL: https://www.postman.com/company/about-postman/
Postman is an API platform for developers and teams to design, build, test,
document, and collaborate on APIs.

[page-2] Postman platform
URL: https://www.postman.com/solutions/api-lifecycle-management/
Postman helps teams manage APIs across the API lifecycle, including design,
testing, documentation, collaboration, and governance.
"""

    extractor = StructuredExtractor()

    print("=" * 70)
    print("LLM STRUCTURED-OUTPUT SMOKE TEST")
    print("=" * 70)
    print(f"Configured provider: {extractor.provider}")
    print(f"Gemini model:       {extractor.gemini_model}")
    print(f"Groq model:         {extractor.groq_model}")
    print()

    result = extractor.extract_overview_icp(evidence)

    print("RESULT")
    print("-" * 70)
    print(f"provider:      {result.provider}")
    print(f"model:         {result.model}")
    print(f"input tokens:  {result.usage.input_tokens}")
    print(f"output tokens: {result.usage.output_tokens}")
    print()
    print(result.value.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
