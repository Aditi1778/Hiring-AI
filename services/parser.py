import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_parse import LlamaParse

from core.config import settings

load_dotenv()


class ResumeParserTool:
    def __init__(self):

        # Settings configure (Global fix)
        self.gemini_llm = Gemini(
            api_key=settings.GEMINI_API_KEY, model_name="models/gemini-2.5-flash"
        )
        self.gemini_embed = GeminiEmbedding(
            api_key=settings.GEMINI_API_KEY, model_name="models/gemini-embedding-001"
        )

        # 2. Force Global Settings
        Settings.llm = self.gemini_llm
        Settings.embed_model = self.gemini_embed
        Settings.chunk_size = 1024  # optimization

        # Initialize LayoutPDFReader with the API URL
        # LlamaParse is now the primary parser for everything
        # 1000 pages per day is more than enough for dev & initial launch
        self.llama_parser = LlamaParse(
            api_key=settings.LLAMA_CLOUD_API_KEY,
            result_type="markdown",  # Markdown preserves layout for better AI matching
            verbose=True,
        )

    def get_smart_index(self, file_path):
        """
        Creates a high-accuracy Vector Index using LlamaParse Cloud.
        Handles PDF, DOCX, and legacy DOC files with layout awareness.
        """

        try:
            # 1. Load data using LlamaParse (Layout-aware parsing)
            # This handles tables, columns, and nested lists perfectly
            original_docs = self.llama_parser.load_data(file_path)

            # # Layout parsing (Columns aur Tables handling)
            # # LLMSherpa automatically detects if it's PDF or DOCX
            # # Background: it sends to the server by converting the file into byte-stream
            # original_docs = self.reader.read_pdf(file_path)

            if not original_docs:
                return None
            # creating smart chunks with header info
            # 2. Create LlamaIndex Documents from Sherpa Chunks
            # to_context_text() ensures headers are attached to each paragraph
            # Hum original docs se text uthayenge aur apna metadata add karenge
            llama_docs = []
            for doc in original_docs:
                llama_docs.append(
                    Document(
                        text=doc.text,  # LlamaParse clean markdown text
                        extra_info={
                            "source": os.path.basename(file_path),
                            "type": "resume_analysis",
                        },
                    )
                )

            if not llama_docs:
                return None

            # Vector Index ready for querying
            # 3. Build the Vector Index (Background: Gemini generates embeddings)
            # semantic search (meaning-based search) gets possible with this
            return VectorStoreIndex.from_documents(
                llama_docs, embed_model=self.gemini_embed, llm=self.gemini_llm
            )

        except Exception as e:
            print(f"LlamaIndex Parsing Error: {e}")
            return None


# Global instance
parser_tool = ResumeParserTool()


# import os
# from pathlib import Path

# from dotenv import load_dotenv
# from llama_index.core import Document, VectorStoreIndex
# from llama_parse import LlamaParse
# from llmsherpa.readers import LayoutPDFReader

# from core.config import settings

# load_dotenv()


# class ResumeParserTool:
#    def __init__(self):
#         # free API server
#         self.api_url = settings.LLMSHERPA_API_URL

#         # Initialize LayoutPDFReader with the API URL
#         self.reader = LayoutPDFReader(self.api_url)

#         # LlamaParse for .doc (Need LLAMA_CLOUD_API_KEY in .env)
#         self.llama_parser = LlamaParse(
#             api_key=settings.LLAMA_CLOUD_API_KEY,
#             result_type="markdown",  # Best for Gemini
#         )

#     def get_smart_index(self, file_path):
#         """
#         Create smart Layout-Aware Index
#         """
#         ext = Path(file_path).suffix.lower()
#         try:
#             if ext == ".doc":
#                 # Background: LlamaParse cloud handles the legacy binary format
#                 llama_docs = self.llama_parser.load_data(file_path)

#             elif ext in [".pdf", ".docx"]:
#                 # Layout parsing (Columns aur Tables handling)
#                 # LLMSherpa automatically detects if it's PDF or DOCX
#                 # Background: it sends to the server by converting the file into byte-stream
#                 doc = self.reader.read_pdf(file_path)
#                 # creating smart chunks with header info
#                 # 2. Create LlamaIndex Documents from Sherpa Chunks
#                 # to_context_text() ensures headers are attached to each paragraph
#                 llama_docs = [
#                     Document(
#                         text=chunk.to_context_text(),
#                         extra_info={"source": os.path.basename(file_path)},
#                     )
#                     for chunk in doc.chunks()
#                 ]

#             else:
#                 return None

#             # Vector Index ready for querying
#             # 3. Build the Vector Index (Background: Gemini generates embeddings)
#             # semantic search (meaning-based search) gets possible with this
#             return VectorStoreIndex.from_documents(llama_docs)

#         except Exception as e:
#             print(f"LlamaIndex Parsing Error: {e}")
#             return None


# # Global instance
# parser_tool = ResumeParserTool()
