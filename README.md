# AI Code Quality Control Tool

This tool performs automated code quality control and validation checks using AWS Bedrock and Claude AI.

## Files

- `qc.py` - Main script that handles AWS Bedrock integration and text generation
- `prompt.txt` - Contains the prompt template for code analysis
- `checklist.txt` - Comprehensive checklist for code review and error prevention

## Setup

1. Create a `.env` file with your AWS credentials:

    ```
    AWS_ACCESS_KEY_ID=your_access_key
    AWS_SECRET_ACCESS_KEY=your_secret_key
    ```

2. Install dependencies:

    ```bash
    pip install boto3 python-dotenv
    ```

## Usage

1. Place the code you want to analyze in a file
2. Run the quality control check:

Basic usage:
```bash
python qc.py --context your_code.py   ####
```


With options:
```bash
python qc.py --context your_code.py --prompt prompt.txt --checklist checklist.txt
```

Options:
- `--context`: File containing the code to analyze
- `--prompt`: Custom prompt template file (default: prompt.txt)
- `--checklist`: Custom checklist file (default: checklist.txt)

The tool will:
- Load the code context
- Apply the checklist criteria
- Generate a detailed analysis in JSON format with:
    - Code issues (with line numbers, severity, and fixes)
    - Quality concerns
    - Security vulnerabilities
    - Performance metrics
    - Overall status and recommendations

## Checklist Categories

- Code Style and Structure
- Error Handling and Runtime Protection
- Input Validation and Security
- Web Scraping and Network Operations
- Data Processing and Storage
- Performance and Optimization
- Testing and Debugging
- Maintenance

## Output Format

```JSON
{
    "status": "success|error",
    "message": "Analysis summary",
    "metadata": {
        "timestamp": "ISO timestamp",
        "version": "analyzer version",
        "execution_time": "time in ms"
    },
    "checks": {
        "code_issues": [
            {
                "line": "line_number",
                "severity": "critical|warning|info",
                "message": "Issue description",
                "error_code": "code snippet",
                "fix": "fix for the issue",
                "code": "fixed code snippet",
                "category": "syntax|logic|security|performance"
            }
        ],
        "quality_issues": [
            {
                "issue": "quality issue",
                "severity": "high|medium|low",
                "fix": "fix for the issues",
                "code": "code snippet",
                "impact": "description of the impact",
                "best_practice": "reference to best practice"
            }
        ],
        "metrics": {
            "complexity": "cyclomatic complexity score",
            "maintainability": "maintainability index",
            "test_coverage": "percentage",
            "duplication": "percentage of duplicate code"
        }
    }
}
```
