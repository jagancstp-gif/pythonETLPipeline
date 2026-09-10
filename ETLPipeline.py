import json
import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Step 1: Load environment variables & initialize OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

INPUT_FILE = "student_enrollment_raw.csv"
OUTPUT_FILE = "cleaned_students.csv"

def clean_batch_with_ai(data_json_string):
    """Passes raw JSON data to OpenAI and returns standardized JSON."""
    system_prompt = """
    You are a data cleaning assistant. You will receive raw student records in JSON format.
    Clean and normalize the dataset based on these rules:
    1. Fix typos, proper casing, and whitespace in Name and City.
    2. Convert Email to lowercase. If the email is invalid (e.g., 'INVALID_EMAIL', missing @, missing domain), set it to null.
    3. Clean Phone to include only digits.
    4. Convert Enrolled_Date to YYYY-MM-DD format.
    5. Ensure Fee_Paid is a strict boolean (true/false).
    
    Return ONLY a valid JSON array of objects with the cleaned records. No explanation or Markdown code block wrapping.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data_json_string}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content.strip()

def run_etl_pipeline():
    # Step 2: Read the file
    print(f"Loading data from {INPUT_FILE}...")
    df_raw = pd.read_csv(INPUT_FILE)
    
    # Convert raw dataframe records to a JSON string for the AI prompt
    raw_json_str = df_raw.to_json(orient="records")

    # Step 3: OpenAI cleans the data
    print("Processing data through OpenAI API...")
    cleaned_json_str = clean_batch_with_ai(raw_json_str)

    # Parse AI JSON output back into a pandas DataFrame
    try:
        cleaned_records = json.loads(cleaned_json_str)
        df_cleaned = pd.DataFrame(cleaned_records)
    except json.JSONDecodeError as e:
        print("Error parsing JSON output from AI:", e)
        return

    # Step 4: Create the final CSV file
    df_cleaned.to_csv(OUTPUT_FILE, index=False)
    print(f"Pipeline finished! Cleaned file saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_etl_pipeline()