# --- Essential N8N Resume Parser FastAPI Server ---

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, Response
import os
import shutil
import tempfile
from datetime import datetime
import logging
import subprocess
from pdf2image import convert_from_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("n8n_resume_parser")

# PDF processing - PyMuPDF is much faster than PyPDF2
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
    logger.info("✅ PyMuPDF available for fast PDF processing")
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("⚠️ PyMuPDF not available, using PyPDF2 fallback")

# Initialize FastAPI app
app = FastAPI(
    title="N8N Resume Parser API",
    description="Essential endpoints for N8N workflow integration",
    version="1.0.0"
)

# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("🚀 Starting N8N Resume Parser API")
    
    # Ensure directories exist
    for folder in ["./documents/uploads", "./documents/temp", "./documents/processed"]:
        os.makedirs(folder, exist_ok=True)
    logger.info("📁 Storage directories initialized")

# --- Essential API Endpoints Used by N8N Workflow ---

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "service": "N8N Resume Parser API",
        "version": "1.0.0",
        "description": "Essential endpoints for N8N workflow integration",
        "workflow_endpoints": {
            "classification_store": "/api/classification/store",
            "document_upload": "/api/documents/upload", 
            "extract_images": "/api/documents/extract-images",
            "extract_text": "/api/documents/extract-text"
        }
    }

@app.post("/api/classification/store")
async def store_classification(request: Request):
    """Store classification results (used by Store Classification1 node)"""
    try:
        data = await request.json()
        logger.info(f"Received classification result: {data}")
        # Simulate storing the classification result
        return {"success": True, "message": "Classification result stored", "data": data}
    except Exception as e:
        logger.error(f"Classification storage failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")

@app.post("/api/documents/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and convert document to PDF (used by HTTP Request node for file upload)"""
    try:
        # Enhanced filename extraction
        received_filename = file.filename
        if not received_filename:
            # Create a temporary filename based on content type
            content_type = file.content_type or 'application/octet-stream'
            if 'pdf' in content_type:
                received_filename = f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            elif 'image' in content_type:
                received_filename = f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            else:
                received_filename = f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin"
        
        logger.info(f"📤 Processing upload: filename='{received_filename}', content_type='{file.content_type}'")
        
        # Validate file
        if not received_filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        # Check file size (max 10MB)
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        if len(file_content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File too large: {file_size_mb:.2f}MB (max: 10MB)")

        # Reset file pointer
        await file.seek(0)

        # Enhanced file type detection
        def detect_document_type(filename: str, content_type: str = None) -> str:
            """Enhanced document type detection"""
            if not filename:
                if content_type:
                    if 'pdf' in content_type.lower():
                        return 'pdf'
                    elif 'image' in content_type.lower():
                        return 'image'
                return 'unknown'
            
            fname_lower = filename.lower()
            if fname_lower.endswith('.pdf'):
                return 'pdf'
            elif fname_lower.endswith('.docx'):
                return 'docx'
            elif fname_lower.endswith('.doc'):
                return 'doc'
            elif fname_lower.endswith(('.jpg', '.jpeg')):
                return 'jpg'
            elif fname_lower.endswith('.png'):
                return 'png'
            elif content_type:
                if 'pdf' in content_type.lower():
                    return 'pdf'
                elif 'image' in content_type.lower():
                    return 'image'
                elif 'word' in content_type.lower():
                    return 'docx' if 'openxml' in content_type.lower() else 'doc'
            
            return 'unknown'

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(received_filename)[1]) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        # Enhanced document type detection
        doc_type = detect_document_type(received_filename, file.content_type)
        converted_path = temp_path
        converted_docx_path = None
        output_pdf_path = None
        
        try:
            # Conversion logic for different document types
            if doc_type == 'docx':
                # Convert DOCX to PDF using LibreOffice (soffice)
                try:
                    pdf_path = os.path.splitext(temp_path)[0] + ".pdf"
                    result = subprocess.run([
                        "soffice", "--headless", "--convert-to", "pdf", temp_path, "--outdir", os.path.dirname(temp_path)
                    ], check=True, capture_output=True, text=True)
                    if os.path.exists(pdf_path):
                        output_pdf_path = pdf_path
                        converted_path = output_pdf_path
                        doc_type = 'pdf'  # After conversion
                        logger.info(f"✅ DOCX converted to PDF using LibreOffice: {pdf_path}")
                    else:
                        logger.error("DOCX to PDF conversion failed - output file not found")
                        logger.warning("Continuing with original DOCX file")
                except Exception as e:
                    logger.error(f"❌ DOCX to PDF conversion failed: {e}")
                    logger.warning("Continuing with original DOCX file")
            
            elif doc_type == 'doc':
                # Convert DOC to DOCX using LibreOffice, then DOCX to PDF
                try:
                    docx_path = os.path.splitext(temp_path)[0] + ".docx"
                    result = subprocess.run([
                        "soffice", "--headless", "--convert-to", "docx", temp_path, "--outdir", os.path.dirname(temp_path)
                    ], check=True, capture_output=True, text=True)
                    if os.path.exists(docx_path):
                        converted_docx_path = docx_path
                        logger.info(f"✅ DOC converted to DOCX: {docx_path}")
                        # Now convert DOCX to PDF
                        try:
                            pdf_path = os.path.splitext(docx_path)[0] + ".pdf"
                            result_pdf = subprocess.run([
                                "soffice", "--headless", "--convert-to", "pdf", docx_path, "--outdir", os.path.dirname(docx_path)
                            ], check=True, capture_output=True, text=True)
                            if os.path.exists(pdf_path):
                                output_pdf_path = pdf_path
                                converted_path = output_pdf_path
                                doc_type = 'pdf'  # After conversion
                                logger.info(f"✅ DOCX converted to PDF using LibreOffice: {pdf_path}")
                            else:
                                logger.error("DOCX to PDF conversion failed - output file not found")
                                converted_path = docx_path
                                doc_type = 'docx'
                        except Exception as e:
                            logger.error(f"❌ DOCX to PDF conversion failed: {e}")
                            converted_path = docx_path
                            doc_type = 'docx'
                    else:
                        logger.error("DOC to DOCX conversion failed - output file not found")
                except Exception as e:
                    logger.error(f"❌ DOC to DOCX conversion failed: {e}")

            logger.info(f"🔍 Final type detection: filename='{received_filename}' → type='{doc_type}'")
                
            logger.info(f"✅ Document processed successfully (type: {doc_type})")
            
            # Return PDF file if conversion successful
            if doc_type == 'pdf' and output_pdf_path and os.path.exists(output_pdf_path):
                background_tasks.add_task(os.unlink, output_pdf_path)
                return FileResponse(
                    output_pdf_path,
                    media_type='application/pdf',
                    filename=os.path.basename(output_pdf_path),
                    background=background_tasks
                )
            # If not a PDF, return error instead of JSON
            logger.error(f"File is not a PDF or PDF conversion failed for: {received_filename}")
            raise HTTPException(status_code=400, detail="File could not be converted to PDF.")
            
        finally:
            # Cleanup temp files
            for temp_file_path in [temp_path, converted_docx_path]:
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except Exception as e:
                        logger.warning(f"Failed to cleanup temp file {temp_file_path}: {e}")
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/documents/extract-images")
async def extract_images_from_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Extract images from PDF (used by Extract Images from PDF node)"""
    try:
        # Save uploaded PDF to temp file
        received_filename = file.filename or f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        logger.info(f"🖼️ Extracting images from PDF: {received_filename}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            shutil.copyfileobj(file.file, temp_pdf)
            temp_pdf_path = temp_pdf.name

        # Convert PDF pages to images with higher DPI for better quality
        images = convert_from_path(temp_pdf_path, dpi=200, fmt='PNG')
        
        # Process first page for LLM analysis
        if images:
            first_page = images[0]
            
            # Save first page as high-quality PNG
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_img:
                first_page.save(temp_img.name, "PNG", optimize=True, quality=95)
                temp_img_path = temp_img.name
            
            # Read image as binary data
            with open(temp_img_path, 'rb') as img_file:
                image_data = img_file.read()
            
            # Schedule cleanup
            background_tasks.add_task(cleanup_temp_files, [temp_pdf_path, temp_img_path])
            
            logger.info(f"✅ Successfully extracted {len(images)} pages, returning first page ({len(image_data)} bytes)")
            
            # Return binary image data for n8n processing
            return Response(
                content=image_data,
                media_type="image/png",
                headers={
                    "X-Original-Filename": received_filename,
                    "X-Page-Count": str(len(images)),
                    "X-Image-Size": str(len(image_data)),
                    "Content-Disposition": f"attachment; filename=page_1_{received_filename.replace('.pdf', '.png')}"
                }
            )
        else:
            # Schedule cleanup
            background_tasks.add_task(cleanup_temp_files, [temp_pdf_path])
            raise HTTPException(status_code=400, detail="No pages found in PDF")
            
    except Exception as e:
        logger.error(f"❌ PDF to images extraction failed: {e}")
        # Clean up temp files on error
        try:
            if 'temp_pdf_path' in locals():
                os.unlink(temp_pdf_path)
            if 'temp_img_path' in locals():
                os.unlink(temp_img_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"PDF to images extraction failed: {str(e)}")

@app.post("/api/documents/extract-text")
async def extract_text_from_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Extract text from PDF (used by Extract Text Fallback node)"""
    try:
        # Save uploaded PDF to temp file
        received_filename = file.filename or f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        logger.info(f"📄 Extracting text from PDF: {received_filename}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            shutil.copyfileobj(file.file, temp_pdf)
            temp_pdf_path = temp_pdf.name

        # Extract text using PyMuPDF (faster) or PyPDF2 (fallback)
        try:
            if PYMUPDF_AVAILABLE:
                # Use PyMuPDF for much faster text extraction
                doc = fitz.open(temp_pdf_path)
                text_content = ""
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                doc.close()
                extraction_method = "pymupdf_fast"
            else:
                # Fallback to PyPDF2
                import PyPDF2
                with open(temp_pdf_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text_content = ""
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                extraction_method = "pypdf2_fallback"
        except Exception as extraction_error:
            logger.error(f"❌ Text extraction failed: {extraction_error}")
            text_content = f"[Text extraction from {received_filename}]\n\nError: {str(extraction_error)}\nPlease ensure the PDF is not corrupted or password-protected."
            extraction_method = "error_fallback"
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, [temp_pdf_path])
        
        logger.info(f"✅ Successfully extracted text ({len(text_content)} characters) using {extraction_method}")
        
        return JSONResponse({
            "success": True,
            "text_content": text_content,
            "character_count": len(text_content),
            "original_filename": received_filename,
            "extraction_method": extraction_method,
            "performance_note": "PyMuPDF provides 5-10x faster extraction than PyPDF2" if extraction_method == "pymupdf_fast" else "Consider installing PyMuPDF for faster processing"
        })
        
    except Exception as e:
        logger.error(f"❌ Text extraction failed: {e}")
        # Clean up temp files on error
        try:
            if 'temp_pdf_path' in locals():
                os.unlink(temp_pdf_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

# --- Helper Functions ---
async def cleanup_temp_files(file_paths: list):
    """Helper function to clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"🗑️ Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup temp file {file_path}: {e}")

# --- Application Entry Point ---
if __name__ == "__main__":
    import uvicorn
    
    # Determine port
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting N8N Resume Parser API")
    logger.info(f"🌐 Server: http://localhost:{port}")
    logger.info(f"📚 API Docs: http://localhost:{port}/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
