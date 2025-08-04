
# --- Imports ---
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request, Body, BackgroundTasks # type: ignore
from fastapi.responses import FileResponse,JSONResponse #type:ignore
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum
import os
import shutil
import tempfile
from datetime import datetime
import logging
import subprocess
from pdf2image import convert_from_path #type:ignore
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("imp")

# --- Enums and Models ---
class ProcessingTier(Enum):
    HIGH_QUALITY = 'high_quality'
    MEDIUM_QUALITY = 'medium_quality'
    LOW_QUALITY = 'low_quality'

class ExtractionTriggerRequest(BaseModel):
    document_id: str
    document_type: str
    sub_type: Optional[str] = None
    confidence: Optional[float] = None
    processing_requirements: Optional[str] = None

class DocumentResponse(BaseModel):
    document_id: str
    original_filename: str
    quality_score: float
    processing_tier: str
    document_type: str
    status: str
    processing_cost: float
    requires_human_review: bool
    processing_time_seconds: float
    ai_mode: bool
    ocr_engine: str = ''
    extracted_fields_count: int = 0
    confidence: float = 0.0

class ProcessingStats(BaseModel):
    total_processed: int
    high_quality_auto: int
    medium_quality_enhanced: int
    low_quality_manual: int
    processing_errors: int
    ai_processed: Optional[int] = None

# --- Initialize components ---

# --- In-memory document metadata store ---
DOCUMENT_STORE: Dict[str, dict] = {}

# --- Minimal stubs for RedisClientStub and DocumentProcessorStub if not defined ---
class RedisClientStub:
    def connect(self):
        return True
    def is_connected(self):
        return True
    def get_cached_document(self, *args, **kwargs):
        return None
    def cache_document_metadata(self, *args, **kwargs):
        pass
    def add_to_queue(self, *args, **kwargs):
        pass
    def increment_counter(self, *args, **kwargs):
        pass
    def get_all_queue_lengths(self):
        return {}
    def get_processing_stats(self):
        return {}
    def get_redis_info(self):
        return {}
    def get_queue_length(self, queue_name: str):
        return 0
    def get_from_queue(self, queue_name: str):
        return None
    def close(self):
        pass
    @property
    def redis_client(self):
        class Dummy:
            def delete(self, *args, **kwargs):
                return 1
        return Dummy()
    CACHE_PREFIXES = {'document': '', 'quality': ''}

class DocumentProcessorStub:
    def process_document(self, file_path, original_filename=None):
        import uuid
        class Result:
            def __init__(self, original_filename=None):
                self.id = str(uuid.uuid4())
                self.original_filename = original_filename if original_filename is not None else 'dummy.pdf'
                class Quality:
                    overall_score = 1.0
                self.quality_metrics = Quality()
                class Tier:
                    value = 'high_quality'
                self.processing_tier = Tier()
                class Status:
                    value = 'completed'
                self.status = Status()
                self.processing_cost = 0.0
                self.ocr_engine = 'none'
                self.extracted_fields_count = 0
                self.confidence = 1.0
        return Result(original_filename)
    def get_processing_stats(self):
        return {}
    def cleanup_resources(self):
        pass

redis_client = RedisClientStub()
document_processor = DocumentProcessorStub()

# Configuration settings
processor_version = ""
ai_mode = True
config = {
    "UPLOAD_FOLDER": "./documents/uploads",
    "TEMP_FOLDER": "./documents/temp",
    "PROCESSED_FOLDER": "./documents/processed",
    "QUALITY_THRESHOLD_HIGH": 0.8,
    "QUALITY_THRESHOLD_MEDIUM": 0.5,
    "QUALITY_THRESHOLD_LOW": 0.2,
    "MAX_FILE_SIZE_MB": 10
}

# Initialize FastAPI app
app = FastAPI(
    title="IMP Document Processing System",
    description="Islamic Manpower Promoters - Smart Document Processing",
    version="1.0.0-"
)

# --- Event Handlers ---
@app.on_event("startup")
async def startup_event():
    """Initialize  application on startup"""
    logger.info(f"🚀 Starting IMP Document Processing System - {processor_version}")
    logger.info(f"📊 AI Mode: {'Enabled' if ai_mode else 'Disabled (Fallback)'}")
    
    # Initialize in-memory storage connection
    storage_connected = redis_client.connect()
    if storage_connected:
        logger.info("💾 In-memory storage initialized - Caching and queues enabled")
    else:
        logger.warning("⚠️ In-memory storage initialization failed")
    
    # Ensure directories exist
    for folder in [config["UPLOAD_FOLDER"], config["TEMP_FOLDER"], config["PROCESSED_FOLDER"]]:
        os.makedirs(folder, exist_ok=True)
    logger.info("📁 Storage directories initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    redis_client.close()
    logger.info("💾 In-memory storage connections closed")
    
    document_processor.cleanup_resources()
    logger.info("🛑 Application shutdown completed")

# --- API Endpoints ---
@app.get("/")
async def root():
    """Root endpoint with  system information"""
    return {
        "service": "IMP Document Processing System",
        "version": "1.0.0-",
        "processor_mode": processor_version,
        "ai_enabled": ai_mode,
        "description": "Islamic Manpower Promoters - Smart Document Processing",
        "endpoints": {
            "upload": "/api/documents/upload",
            "extract": "/api/documents/extract",
            "status": "/api/documents/{id}/status",
            "stats": "/api/processing/stats",
            "health": "/api/health"
        },
        "features": [
            "Smart document classification",
            "Quality-based processing tiers",
            "AI-powered OCR (when available)",
            "Automatic fallback to basic processing",
            "Cost optimization",
            "Human-in-the-loop for low quality documents"
        ]
    }

@app.get("/api/health")
async def health_check():
    """Enhanced health check with storage and system status"""
    try:
        # Test document processor
        stats = document_processor.get_processing_stats()
        
        # Check AI availability
        ai_status = "available" if ai_mode else "unavailable (fallback active)"
        
        # Check storage status
        storage_connected = redis_client.is_connected()
        storage_status = "connected" if storage_connected else "disconnected"
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "processor_mode": processor_version,
            "ai_mode": ai_mode,
            "ai_status": ai_status,
            "storage_connected": storage_connected,
            "storage_status": storage_status,
            "total_processed": stats.get('total_processed', 0),
            "error_rate": stats.get('processing_errors', 0) / max(stats.get('total_processed', 1), 1) * 100,
            "version": "1.0.0-"
        }
        
        # Add queue info if available
        if storage_connected:
            try:
                queue_lengths = redis_client.get_all_queue_lengths()
                health_data["queue_lengths"] = queue_lengths
                health_data["total_queued"] = sum(queue_lengths.values())
            except Exception:
                health_data["queue_status"] = "limited_access"
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/api/classification/store")
async def store_classification(request: Request):
    """Store classification results (Phase 1: Simulated storage)"""
    try:
        data = await request.json()
        logger.info(f"Received classification result: {data}")
        # Simulate storing the classification result
        return {"success": True, "message": "Classification result stored", "data": data}
    except Exception as e:
        logger.error(f"Classification storage failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")

@app.post("/api/extraction/trigger")
async def extraction_trigger(payload: ExtractionTriggerRequest):
    """Stub endpoint to trigger extraction (for n8n workflow integration)"""
    logger.info(f"Extraction trigger received: {payload}")
    return {
        "success": True,
        "message": "Extraction trigger received",
        **payload.dict()
    }

@app.post("/api/documents/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process document with enhanced filename handling"""
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

        # Check file size
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        if len(file_content) > config["MAX_FILE_SIZE_MB"] * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File too large: {file_size_mb:.2f}MB (max: {config['MAX_FILE_SIZE_MB']}MB)")

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

        # Check memory cache first (if available)
        file_hash = f"{received_filename}_{len(file_content)}_{datetime.now().strftime('%Y%m%d')}"
        cached_result = redis_client.get_cached_document(file_hash)
        if cached_result:
            logger.info(f"📋 Returning cached result for {received_filename}")
            return DocumentResponse(**cached_result)

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
            # Conversion logic (same as before but with better error handling)
            if doc_type == 'docx':
                # Convert DOCX to PDF using LibreOffice (soffice) for Linux compatibility
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
                # Convert DOC to DOCX using LibreOffice (soffice), then DOCX to PDF using LibreOffice
                try:
                    docx_path = os.path.splitext(temp_path)[0] + ".docx"
                    result = subprocess.run([
                        "soffice", "--headless", "--convert-to", "docx", temp_path, "--outdir", os.path.dirname(temp_path)
                    ], check=True, capture_output=True, text=True)
                    if os.path.exists(docx_path):
                        converted_docx_path = docx_path
                        logger.info(f"✅ DOC converted to DOCX: {docx_path}")
                        # Now convert DOCX to PDF using LibreOffice
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
                    # Continue with original file

            logger.info(f"🔍 Final type detection: filename='{received_filename}' → type='{doc_type}'")

            # Process document with processor (use converted_path)
            result = document_processor.process_document(
                converted_path,
                original_filename=received_filename
            )

            # Build enhanced response data
            response_data = {
                "document_id": result.id,
                "original_filename": received_filename,  # Always preserve original filename
                "quality_score": result.quality_metrics.overall_score,
                "processing_tier": result.processing_tier.value,
                "document_type": doc_type,
                "status": result.status.value,
                "processing_cost": result.processing_cost,
                "requires_human_review": result.processing_tier == ProcessingTier.LOW_QUALITY,
                "processing_time_seconds": 0.0,
                "ai_mode": ai_mode,
                "ocr_engine": result.ocr_engine,
                "extracted_fields_count": result.extracted_fields_count,
                "confidence": result.confidence,
                # Additional metadata for n8n workflow
                "file_size_mb": round(file_size_mb, 2),
                "content_type": file.content_type,
                "converted_docx": os.path.basename(converted_docx_path) if converted_docx_path else None,
                "converted_pdf": os.path.basename(output_pdf_path) if output_pdf_path else None,
                # Debugging information
                "processing_metadata": {
                    "original_type": detect_document_type(received_filename, file.content_type),
                    "final_type": doc_type,
                    "conversion_performed": output_pdf_path is not None or converted_docx_path is not None,
                    "file_hash": file_hash
                }
            }

            # Cache successful results in memory (if available)
            redis_client.cache_document_metadata(file_hash, response_data, ttl=3600)
            # Store metadata in DOCUMENT_STORE for retrieval
            DOCUMENT_STORE[result.id] = response_data.copy()

            # Add to processing queue based on quality
            queue_data = {
                'document_id': result.id,
                'quality_score': response_data["quality_score"],
                'processing_tier': result.processing_tier.value,
                'original_filename': received_filename,
                'document_type': doc_type
            }

            if result.processing_tier == ProcessingTier.HIGH_QUALITY:
                redis_client.add_to_queue('high_priority', queue_data)
                redis_client.increment_counter('processing:high_quality')
            elif result.processing_tier == ProcessingTier.MEDIUM_QUALITY:
                redis_client.add_to_queue('medium_priority', queue_data)
                redis_client.increment_counter('processing:medium_quality')
            else:
                queue_data['requires_human_review'] = True
                redis_client.add_to_queue('human_review', queue_data)
                redis_client.increment_counter('processing:low_quality')

            # Update general counters
            redis_client.increment_counter('documents:processed')
                
            logger.info(f"✅ Document processed successfully: {result.id} (type: {doc_type})")
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
            # Only delete temp_path and converted_docx_path here.
            # output_pdf_path is deleted by background_tasks if sent as response.
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
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")\
# --- PDF to Images Endpoint ---


@app.post("/api/documents/extract-images")
async def extract_images_from_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Extract images (pages as images) from a PDF and return file names."""
    try:
        # Save uploaded PDF to temp file
        received_filename = file.filename or f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            shutil.copyfileobj(file.file, temp_pdf)
            temp_pdf_path = temp_pdf.name

        # Output directory for images
        output_dir = os.path.join(config["TEMP_FOLDER"], f"pdf_images_{uuid.uuid4().hex}")
        os.makedirs(output_dir, exist_ok=True)

        # Convert PDF pages to images
        images = convert_from_path(temp_pdf_path)
        image_files = []
        for i, img in enumerate(images):
            img_filename = f"page_{i+1}.png"
            img_path = os.path.join(output_dir, img_filename)
            img.save(img_path, "PNG")
            image_files.append(img_path)

        # Optionally, schedule cleanup of temp files
        background_tasks.add_task(os.unlink, temp_pdf_path)
        # Optionally, schedule cleanup of output_dir and images after some time

        # Return list of image file paths (relative or absolute)
        return JSONResponse({
            "success": True,
            "image_files": image_files,
            "output_dir": output_dir,
            "page_count": len(image_files)
        })
    except Exception as e:
        logger.error(f"❌ PDF to images extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF to images extraction failed: {str(e)}")
@app.get("/api/documents/{document_id}/status")
async def get_document_status(document_id: str):
    """Get processing status for a document"""
    try:
        return {
            "document_id": document_id,
            "status": "completed",
            "processor_mode": processor_version,
            "ai_mode": ai_mode,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Status check failed: {e}")
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")


@app.get("/api/documents/{document_id}")
async def get_document(document_id: str):
    """Get document details by ID (real, from in-memory store)"""
    try:
        logger.info(f"Fetching document metadata for ID: {document_id}")
        logger.info(f"Current DOCUMENT_STORE keys: {list(DOCUMENT_STORE.keys())}")
        doc = DOCUMENT_STORE.get(document_id)
        if not doc:
            logger.warning(f"Document ID {document_id} not found in DOCUMENT_STORE.")
            raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")
        logger.info(f"Found document metadata: {doc}")
        return DocumentResponse(**doc)
    except Exception as e:
        logger.error(f"❌ Document retrieval failed for ID {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Document retrieval failed: {str(e)}")

@app.get("/api/processing/stats", response_model=ProcessingStats)
async def get_processing_statistics():
    """processing statistics"""
    try:
        stats = document_processor.get_processing_stats()
        
        response_data = {
            "total_processed": stats.get('total_processed', 0),
            "high_quality_auto": stats.get('high_quality_auto', 0),
            "medium_quality_enhanced": stats.get('medium_quality_enhanced', 0),
            "low_quality_manual": stats.get('low_quality_manual', 0),
            "processing_errors": stats.get('processing_errors', 0),
            "ai_mode_active": ai_mode
        }
        
        # Add AI-specific stats if available
        if ai_mode:
            response_data["ai_processed"] = stats.get('donut_ocr_processed', 0)
            response_data["donut_ocr_processed"] = stats.get('donut_ocr_processed', 0)
        
        return ProcessingStats(**response_data)
        
    except Exception as e:
        logger.error(f"❌ Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stats unavailable: {str(e)}")

@app.get("/api/config")
async def get_configuration():
    """Get current system configuration"""
    return {
        "processor_mode": processor_version,
        "ai_enabled": ai_mode,
        "quality_thresholds": {
            "high": config["QUALITY_THRESHOLD_HIGH"],
            "medium": config["QUALITY_THRESHOLD_MEDIUM"],
            "low": config["QUALITY_THRESHOLD_LOW"]
        },
        "file_limits": {
            "max_size_mb": config["MAX_FILE_SIZE_MB"]
        },
        "storage": {
            "upload_folder": config["UPLOAD_FOLDER"],
            "temp_folder": config["TEMP_FOLDER"],
            "processed_folder": config["PROCESSED_FOLDER"]
        }
    }

@app.get("/api/storage/stats")
async def get_storage_stats():
    """Get in-memory storage statistics and queue information"""
    try:
        if not redis_client.is_connected():
            return {"message": "Storage not available", "stats": {}}
        
        # Get comprehensive storage statistics
        storage_stats = redis_client.get_processing_stats()
        storage_info = redis_client.get_redis_info()
        
        return {
            "storage_connected": True,
            "processing_stats": storage_stats,
            "storage_info": storage_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Storage stats failed: {e}")
        return {"storage_connected": False, "error": str(e)}

@app.get("/api/storage/queues")
async def get_queue_status():
    """Get current queue lengths and status"""
    try:
        if not redis_client.is_connected():
            return {"message": "Storage not available", "queues": {}}
        
        queue_lengths = redis_client.get_all_queue_lengths()
        
        return {
            "storage_connected": True,
            "queues": queue_lengths,
            "total_queued": sum(queue_lengths.values()),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Queue status failed: {e}")
        return {"storage_connected": False, "error": str(e)}

@app.get("/api/redis/queue/{queue_name}")
async def get_queue_details(queue_name: str):
    """Get specific queue details and pending items"""
    try:
        if not redis_client.is_connected():
            return {"message": "Redis not available"}
        
        queue_length = redis_client.get_queue_length(queue_name)
        
        return {
            "queue_name": queue_name,
            "length": queue_length,
            "redis_connected": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Queue details failed: {e}")
        return {"error": str(e), "redis_connected": False}

@app.post("/api/redis/queue/{queue_name}/process")
async def process_queue_item(queue_name: str):
    """Process next item from specified queue"""
    try:
        if not redis_client.is_connected():
            raise HTTPException(status_code=503, detail="Redis not available")
        
        # Get next item from queue
        item = redis_client.get_from_queue(queue_name)
        
        if not item:
            return {"message": f"No items in {queue_name} queue", "processed": False}
        
        # Process the item (basic implementation)
        document_id = item.get('document_id')
        logger.info(f"🔄 Processing queue item: {document_id} from {queue_name}")
        
        # Update counters
        redis_client.increment_counter(f"queue:{queue_name}:processed")
        
        return {
            "message": f"Processed item from {queue_name}",
            "document_id": document_id,
            "item_data": item,
            "processed": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Queue processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Queue processing failed: {str(e)}")

@app.delete("/api/redis/cache/{document_id}")
async def clear_document_cache(document_id: str):
    """Clear cached data for a specific document"""
    try:
        if not redis_client.is_connected():
            raise HTTPException(status_code=503, detail="Redis not available")
        
        # Clear document cache
        doc_deleted = redis_client.redis_client.delete(f"{redis_client.CACHE_PREFIXES['document']}{document_id}")
        quality_deleted = redis_client.redis_client.delete(f"{redis_client.CACHE_PREFIXES['quality']}{document_id}")
        
        return {
            "document_id": document_id,
            "cache_cleared": True,
            "items_deleted": doc_deleted + quality_deleted,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Cache clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

# Run the application
if __name__ == "__main__":
    import uvicorn  #type:ignore 
    
    # Determine port and mode
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting  IMP Document Processing System")
    logger.info(f"📊 Mode: {processor_version}")
    logger.info(f"🌐 Server: http://localhost:{port}")
    logger.info(f"📚 API Docs: http://localhost:{port}/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Set to True for development
        log_level="info"
    )