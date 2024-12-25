import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pdf2image
import pytesseract
import re
import spacy
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os
from datetime import datetime
import json
import tempfile
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Person:
    name: str
    gov_id: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None

@dataclass
class Document:
    type: str
    content: str
    confidence: float
    metadata: Dict
    person: Optional[Person] = None
    raw_text: str = ""
    processed_date: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class DocumentDatabase:
    def __init__(self, db_path: str = "documents.json"):
        self.db_path = db_path
        self.documents = self._load_db()

    def _load_db(self) -> Dict:
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return {}

    def save_document(self, doc_id: str, document: Document):
        self.documents[doc_id] = {
            'type': document.type,
            'person': document.person.__dict__ if document.person else None,
            'confidence': document.confidence,
            'metadata': document.metadata,
            'processed_date': document.processed_date
        }
        with open(self.db_path, 'w') as f:
            json.dump(self.documents, f)

class DocumentProcessor:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found")
            
        self.nlp = spacy.load("en_core_web_sm")
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        self.db = DocumentDatabase()
        
        self.doc_patterns = {
            "bank_account_application": [
                r"account\s*application",
                r"new\s*account",
                r"account\s*opening",
            ],
            "identity_document": [
                r"passport",
                r"driver'?s?\s*license",
                r"identification\s*card"
            ],
            "financial_document": [
                r"pay\s*stub",
                r"income\s*statement",
                r"tax\s*return",
            ],
            "receipt": [
                r"receipt",
                r"invoice",
                r"payment\s*confirmation"
            ]
        }

    def extract_text(self, pdf_path: str) -> Tuple[str, float]:
        try:
            images = pdf2image.convert_from_path(pdf_path)
            text = ""
            quality_scores = []
            
            for img in images:
                data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                page_text = " ".join([word for word in data['text'] if word.strip()])
                confidence = sum(float(conf) for conf in data['conf'] if conf != '-1') / len(data['conf'])
                
                text += page_text + "\n"
                quality_scores.append(confidence)
            
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            return text.strip(), avg_quality / 100
            
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            return "", 0.0

    def extract_person_info(self, text: str) -> Person:
        doc = self.nlp(text)
        
        names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
        address_pattern = r'\d+\s+[A-Za-z0-9\s,]+\b(?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|parkway|pkwy|circle|cir|boulevard|blvd)\b'
        gov_id_pattern = r'(?i)(?:ID|DL|SSN|passport)\s*#?\s*([A-Z0-9-]+)'
        
        return Person(
            name=names[0] if names else "",
            gov_id=re.findall(gov_id_pattern, text)[0] if re.findall(gov_id_pattern, text) else None,
            email=re.findall(email_pattern, text)[0] if re.findall(email_pattern, text) else None,
            phone=re.findall(phone_pattern, text)[0] if re.findall(phone_pattern, text) else None,
            address=re.findall(address_pattern, text, re.IGNORECASE)[0] if re.findall(address_pattern, text, re.IGNORECASE) else None
        )

    def classify_document(self, text: str) -> Tuple[str, float]:
        text_lower = text.lower()
        max_confidence = 0.0
        doc_type = "unknown"
        
        for type_name, patterns in self.doc_patterns.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, text_lower))
            confidence = matches / len(patterns)
            if confidence > max_confidence:
                max_confidence = confidence
                doc_type = type_name
        
        if max_confidence < 0.5:
            try:
                prompt = f"Classify this document into: bank_account_application, identity_document, financial_document, receipt\n\nText: {text[:1000]}..."
                llm_classification = self.llm.predict(prompt).strip().lower()
                if llm_classification in self.doc_patterns:
                    doc_type = llm_classification
                    max_confidence = 0.7
            except Exception as e:
                logger.error(f"LLM classification error: {e}")
        
        return doc_type, max_confidence

    def extract_metadata(self, text: str, doc_type: str) -> Dict:
        metadata = {
            "dates": [],
            "amounts": [],
            "account_numbers": [],
            "important_entities": []
        }
        
        date_patterns = [r'\d{2}/\d{2}/\d{4}', r'\d{2}-\d{2}-\d{4}']
        amount_pattern = r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?'
        account_pattern = r'(?i)(?:account|acct|a/c).*?([0-9X]{8,})'
        
        metadata["dates"] = [d for pattern in date_patterns for d in re.findall(pattern, text)]
        metadata["amounts"] = re.findall(amount_pattern, text)
        metadata["account_numbers"] = re.findall(account_pattern, text)
        
        doc = self.nlp(text)
        metadata["important_entities"] = [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
            if ent.label_ in ["ORG", "GPE", "MONEY", "PERCENT"]
        ]
        
        return metadata

    def process_document(self, pdf_path: str) -> Document:
        text, ocr_quality = self.extract_text(pdf_path)
        doc_type, classification_confidence = self.classify_document(text)
        person = self.extract_person_info(text)
        metadata = self.extract_metadata(text, doc_type)
        
        person_info_score = sum([
            bool(person.name) * 0.4,
            bool(person.gov_id) * 0.2,
            bool(person.email) * 0.2,
            bool(person.address) * 0.2
        ])
        
        overall_confidence = (
            ocr_quality * 0.3 +
            classification_confidence * 0.3 +
            person_info_score * 0.4
        )
        
        return Document(
            type=doc_type,
            content=text[:1000],
            confidence=overall_confidence,
            metadata=metadata,
            person=person,
            raw_text=text
        )

def create_document_summary(doc: Document) -> str:
    return f"""
Document Type: {doc.type} (Confidence: {doc.confidence:.2f})
Processed Date: {doc.processed_date}

Person Information:
- Name: {doc.person.name if doc.person else 'Not found'}
- ID: {doc.person.gov_id if doc.person and doc.person.gov_id else 'Not found'}
- Email: {doc.person.email if doc.person and doc.person.email else 'Not found'}
- Phone: {doc.person.phone if doc.person and doc.person.phone else 'Not found'}
- Address: {doc.person.address if doc.person and doc.person.address else 'Not found'}

Metadata:
- Dates: {', '.join(doc.metadata['dates'][:3])}
- Amounts: {', '.join(doc.metadata['amounts'][:3])}
- Account Numbers: {', '.join(doc.metadata['account_numbers'][:3])}

Important Entities:
{chr(10).join([f"- {ent['text']} ({ent['label']})" for ent in doc.metadata['important_entities'][:5]])}
    """.strip()