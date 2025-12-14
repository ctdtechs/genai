import os
import streamlit as st
import pdfplumber
from PIL import Image
import json
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Document Processor",
    layout="wide"
)

MODEL_ID = "apac.anthropic.claude-3-5-sonnet-20241022-v2:0"

AWS_REGION = st.secrets.get("AWS_REGION", "ap-south-1")
AWS_ACCESS_KEY = st.secrets.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = st.secrets.get("AWS_SECRET_ACCESS_KEY")


def get_bedrock_client():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )


def converse(brt, model_id, user_message):
    try:
        response = brt.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": user_message}]
                }
            ],
            inferenceConfig={
                "maxTokens": 8192,
                "temperature": 0.5,
                "topP": 0.9
            }
        )
        return response["output"]["message"]["content"][0]["text"]
    except (ClientError, Exception) as e:
        raise RuntimeError(f"Bedrock invocation failed: {e}")


def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def process_image(file):
    image = Image.open(file)
    return {
        "document_type": "Image",
        "format": image.format,
        "mode": image.mode,
        "width": image.width,
        "height": image.height
    }


def build_prompt(document_text):
    return f"""
You are an AI document intelligence assistant.

TASKS:
1. Generate a concise business-friendly summary.
2. Identify the document domain (Healthcare, Legal, Finance, Insurance, Government, etc.).
3. Identify document origin/geography (India, US, EU, etc.) if possible.
4. Identify document type.
5. Decide ERP readiness.
6. Transform the data into structured JSON.

ERP READINESS RULES:
- READY: Key fields are clear and usable
- NOT_READY: Missing or unclear fields

STRICT OUTPUT FORMAT (JSON ONLY):
{{
  "summary": "<short summary>",
  "domain": "<Healthcare | Legal | Finance | Insurance | Other>",
  "origin": "<Country or Region or Unknown>",
  "document_type": "<Invoice | Judgment | Medical Report | Policy | Other>",
  "erp_status": "<READY | NOT_READY>",
  "transformed_data": {{
    "key_points": [],
    "entities": {{}},
    "confidence": 0.0
  }}
}}

<Input_Text>
{document_text}
</Input_Text>
"""


st.title("📄 Document Processor")
st.caption("Standalone Streamlit app | Summary → ERP Decision → Transformed Data")

uploaded_file = st.file_uploader(
    "Upload PDF or Image",
    type=["pdf", "png", "jpg", "jpeg"]
)

# --------------------------------------------------
# PROCESS
# --------------------------------------------------

if uploaded_file:

    st.info(f"Uploaded file: **{uploaded_file.name}**")

    if st.button("🚀 Process Document"):

        with st.spinner("Processing document using CTD's Gen AI..."):

            extracted_text = ""

            if uploaded_file.type == "application/pdf":
                extracted_text = extract_text_from_pdf(uploaded_file)

            elif uploaded_file.type.startswith("image/"):
                image_info = process_image(uploaded_file)
                extracted_text = json.dumps(image_info, indent=2)

            else:
                st.error("Unsupported file type")
                st.stop()

            if not extracted_text:
                st.error("No content extracted")
                st.stop()

            prompt = build_prompt(extracted_text)

            try:
                brt = get_bedrock_client()
                llm_response = converse(brt, MODEL_ID, prompt)
                parsed = json.loads(llm_response)
            except Exception as e:
                st.error("Failed to process document")
                st.text(str(e))
                st.text(llm_response if "llm_response" in locals() else "")
                st.stop()

        # --------------------------------------------------
        # SUMMARY FIRST
        # --------------------------------------------------
        st.subheader("📌 Document Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Domain", parsed.get("domain", "Unknown"))

        with col2:
            st.metric("Origin", parsed.get("origin", "Unknown"))

        with col3:
            st.metric("Document Type", parsed.get("document_type", "Unknown"))

        st.success(parsed.get("summary", "No summary available"))

        # --------------------------------------------------
        # ERP READINESS
        # --------------------------------------------------
        st.subheader("⚙️ ERP Readiness")

        erp_status = parsed.get("erp_status", "NOT_READY")

        if erp_status == "READY":
            st.success("✅ Ready to Load ERP Service")
        else:
            st.warning("⏳ Needs Manual Review Before ERP Load")

        colA, colB = st.columns(2)

        with colA:
            st.button("🚀 Send to ERP")

        with colB:
            st.button("📝 Mark for Manual Review")

        # --------------------------------------------------
        # DETAILS
        # --------------------------------------------------
        tab1, tab2 = st.tabs(["📝 Extracted Text", "📊 Transformed Data"])

        with tab1:
            st.text_area(
                "Extracted Content",
                extracted_text,
                height=400
            )

        with tab2:
            transformed_data = parsed.get("transformed_data", {})
            st.json(transformed_data)

            st.download_button(
                "⬇ Download Transformed JSON",
                data=json.dumps(transformed_data, indent=2),
                file_name="transformed_data.json",
                mime="application/json"
            )

