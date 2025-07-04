-- IMP Document Processing System Database Initialization
-- This script creates the necessary tables and indexes for the system

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS n8n;
CREATE SCHEMA IF NOT EXISTS candidate_management;
CREATE SCHEMA IF NOT EXISTS job_management;
CREATE SCHEMA IF NOT EXISTS document_processing;

-- Set default schema for subsequent operations
SET search_path = candidate_management, job_management, document_processing, public;

-- =============================================
-- USERS AND AUTHENTICATION (Document Processing)
-- =============================================

CREATE TABLE document_processing.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'reviewer',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITHOUT TIME ZONE,
    
    CONSTRAINT users_role_check CHECK (role IN ('admin', 'supervisor', 'reviewer'))
);

-- =============================================
-- JOB MANAGEMENT SCHEMA
-- =============================================

CREATE TABLE job_management.skills (
    id SERIAL PRIMARY KEY,
    skill_name VARCHAR(255) NOT NULL UNIQUE,
    category VARCHAR(100),
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE job_management.jobs (
    id SERIAL PRIMARY KEY,
    job_title VARCHAR(255) NOT NULL,
    job_code VARCHAR(100) UNIQUE NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    location VARCHAR(255),
    country VARCHAR(100),
    job_type VARCHAR(50), -- Full-time, Part-time, Contract
    salary_min NUMERIC,
    salary_max NUMERIC,
    currency VARCHAR(10) DEFAULT 'USD',
    description TEXT,
    requirements TEXT,
    status VARCHAR(50) DEFAULT 'Active',
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- CANDIDATE MANAGEMENT SCHEMA
-- =============================================

-- Create sequences first
CREATE SEQUENCE candidate_management.candidates_id_seq;
CREATE SEQUENCE candidate_management.educations_id_seq;
CREATE SEQUENCE candidate_management.experiences_id_seq;
CREATE SEQUENCE candidate_management.candidate_skills_id_seq;
CREATE SEQUENCE candidate_management.job_applications_id_seq;

-- Create ENUM types
CREATE TYPE candidate_management."ApplicationStatus" AS ENUM (
    'APPLIED', 'SCREENING', 'INTERVIEW', 'OFFERED', 'HIRED', 'REJECTED', 'WITHDRAWN'
);

CREATE TABLE candidate_management.candidates (
    id INTEGER NOT NULL DEFAULT nextval('candidate_management.candidates_id_seq'),
    candidate_code VARCHAR NOT NULL UNIQUE,
    first_name VARCHAR NOT NULL,
    last_name VARCHAR NOT NULL,
    email VARCHAR,
    phone VARCHAR,
    nationality VARCHAR,
    date_of_birth TIMESTAMP WITHOUT TIME ZONE,
    gender VARCHAR,
    marital_status VARCHAR,
    address TEXT,
    city VARCHAR,
    state VARCHAR,
    country VARCHAR,
    postal_code VARCHAR,
    passport_number VARCHAR,
    passport_expiry TIMESTAMP WITHOUT TIME ZONE,
    visa_status VARCHAR,
    current_salary NUMERIC,
    expected_salary NUMERIC,
    availability_date TIMESTAMP WITHOUT TIME ZONE,
    status VARCHAR NOT NULL DEFAULT 'Active',
    source VARCHAR,
    notes TEXT,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Document processing specific fields
    processing_stage VARCHAR(100) DEFAULT 'uploaded',
    document_quality_score INTEGER DEFAULT 0,
    
    CONSTRAINT candidates_pkey PRIMARY KEY (id),
    CONSTRAINT candidates_document_quality_score_check CHECK (document_quality_score >= 0 AND document_quality_score <= 100)
);

CREATE TABLE candidate_management.educations (
    id INTEGER NOT NULL DEFAULT nextval('candidate_management.educations_id_seq'),
    candidate_id INTEGER NOT NULL,
    institution_name VARCHAR NOT NULL,
    degree VARCHAR NOT NULL,
    field_of_study VARCHAR,
    start_date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    end_date TIMESTAMP WITHOUT TIME ZONE,
    gpa VARCHAR,
    country VARCHAR,
    is_completed BOOLEAN NOT NULL DEFAULT true,
    
    CONSTRAINT educations_pkey PRIMARY KEY (id),
    CONSTRAINT educations_candidate_id_fkey FOREIGN KEY (candidate_id) REFERENCES candidate_management.candidates(id) ON DELETE CASCADE
);

CREATE TABLE candidate_management.experiences (
    id INTEGER NOT NULL DEFAULT nextval('candidate_management.experiences_id_seq'),
    candidate_id INTEGER NOT NULL,
    company_name VARCHAR NOT NULL,
    job_title VARCHAR NOT NULL,
    job_description TEXT,
    start_date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    end_date TIMESTAMP WITHOUT TIME ZONE,
    is_current BOOLEAN NOT NULL DEFAULT false,
    salary NUMERIC,
    currency VARCHAR,
    country VARCHAR,
    achievements TEXT,
    reason_for_leaving TEXT,
    
    CONSTRAINT experiences_pkey PRIMARY KEY (id),
    CONSTRAINT experiences_candidate_id_fkey FOREIGN KEY (candidate_id) REFERENCES candidate_management.candidates(id) ON DELETE CASCADE
);

CREATE TABLE candidate_management.candidate_skills (
    id INTEGER NOT NULL DEFAULT nextval('candidate_management.candidate_skills_id_seq'),
    candidate_id INTEGER NOT NULL,
    skill_id INTEGER NOT NULL,
    proficiency_level VARCHAR NOT NULL,
    years_of_experience INTEGER,
    certified BOOLEAN NOT NULL DEFAULT false,
    certification_name VARCHAR,
    last_used TIMESTAMP WITHOUT TIME ZONE,
    
    CONSTRAINT candidate_skills_pkey PRIMARY KEY (id),
    CONSTRAINT candidate_skills_candidate_id_fkey FOREIGN KEY (candidate_id) REFERENCES candidate_management.candidates(id) ON DELETE CASCADE,
    CONSTRAINT candidate_skills_skill_id_fkey FOREIGN KEY (skill_id) REFERENCES job_management.skills(id)
);

CREATE TABLE candidate_management.job_applications (
    id INTEGER NOT NULL DEFAULT nextval('candidate_management.job_applications_id_seq'),
    application_number VARCHAR NOT NULL UNIQUE,
    job_id INTEGER NOT NULL,
    candidate_id INTEGER NOT NULL,
    application_date TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status candidate_management."ApplicationStatus" NOT NULL DEFAULT 'APPLIED',
    stage VARCHAR NOT NULL DEFAULT 'Initial',
    priority VARCHAR NOT NULL DEFAULT 'Normal',
    source VARCHAR,
    applied_salary NUMERIC,
    notes TEXT,
    rejection_reason TEXT,
    processed_by INTEGER,
    processed_at TIMESTAMP WITHOUT TIME ZONE,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT job_applications_pkey PRIMARY KEY (id),
    CONSTRAINT job_applications_job_id_fkey FOREIGN KEY (job_id) REFERENCES job_management.jobs(id),
    CONSTRAINT job_applications_candidate_id_fkey FOREIGN KEY (candidate_id) REFERENCES candidate_management.candidates(id) ON DELETE CASCADE
);

-- =============================================
-- DOCUMENT PROCESSING SCHEMA
-- =============================================

CREATE TABLE document_processing.documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    candidate_id INTEGER REFERENCES candidate_management.candidates(id) ON DELETE CASCADE,
    document_type VARCHAR(100) NOT NULL, -- cv, passport, certificate, etc.
    original_filename VARCHAR(500) NOT NULL,
    file_path VARCHAR(1000) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    quality_score INTEGER DEFAULT 0, -- 0-100 document quality
    processing_status VARCHAR(50) DEFAULT 'uploaded',
    extracted_data JSONB, -- Structured extracted data
    extraction_confidence JSONB, -- Confidence scores per field
    ocr_engines_used TEXT[], -- Array of OCR engines used
    processing_errors TEXT[],
    uploaded_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITHOUT TIME ZONE,
    
    CONSTRAINT documents_quality_score_check CHECK (quality_score >= 0 AND quality_score <= 100),
    CONSTRAINT documents_processing_status_check CHECK (processing_status IN ('uploaded', 'processing', 'completed', 'failed', 'review_required'))
);

CREATE TABLE document_processing.processing_queue (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES document_processing.documents(id) ON DELETE CASCADE,
    workflow_type VARCHAR(100) NOT NULL, -- auto, enhanced, manual_review
    priority INTEGER DEFAULT 5, -- 1-10, higher = more priority
    assigned_to UUID REFERENCES document_processing.users(id),
    status VARCHAR(50) DEFAULT 'pending',
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    scheduled_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITHOUT TIME ZONE,
    completed_at TIMESTAMP WITHOUT TIME ZONE,
    error_message TEXT,
    workflow_data JSONB, -- Additional workflow context
    
    CONSTRAINT processing_queue_status_check CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    CONSTRAINT processing_queue_priority_check CHECK (priority >= 1 AND priority <= 10)
);

CREATE TABLE document_processing.verification_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES document_processing.documents(id) ON DELETE CASCADE,
    assigned_to UUID REFERENCES document_processing.users(id),
    task_type VARCHAR(100) NOT NULL, -- quality_review, data_correction, manual_entry
    original_data JSONB, -- AI extracted data
    corrected_data JSONB, -- Human corrected data
    verification_notes TEXT,
    confidence_override JSONB, -- Manual confidence adjustments
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITHOUT TIME ZONE,
    completed_at TIMESTAMP WITHOUT TIME ZONE,
    time_spent_minutes INTEGER,
    
    CONSTRAINT verification_tasks_status_check CHECK (status IN ('pending', 'in_progress', 'completed', 'rejected'))
);

CREATE TABLE document_processing.api_usage_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_provider VARCHAR(100) NOT NULL, -- huggingface, google_vision, openai
    api_endpoint VARCHAR(255) NOT NULL,
    document_id UUID REFERENCES document_processing.documents(id),
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    processing_time_ms INTEGER,
    cost_usd DECIMAL(10,6),
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT api_usage_logs_cost_check CHECK (cost_usd >= 0)
);

CREATE TABLE document_processing.system_settings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value JSONB NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE document_processing.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES document_processing.users(id),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL, -- candidate, document, user
    resource_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- CREATE INDEXES FOR PERFORMANCE
-- =============================================

-- Candidate Management indexes
CREATE INDEX idx_candidates_status ON candidate_management.candidates(status);
CREATE INDEX idx_candidates_processing_stage ON candidate_management.candidates(processing_stage);
CREATE INDEX idx_candidates_created_at ON candidate_management.candidates(created_at);
CREATE INDEX idx_candidates_candidate_code ON candidate_management.candidates(candidate_code);
CREATE INDEX idx_candidates_first_name ON candidate_management.candidates(first_name);
CREATE INDEX idx_candidates_last_name ON candidate_management.candidates(last_name);
CREATE INDEX idx_candidates_email ON candidate_management.candidates(email);

-- Education indexes
CREATE INDEX idx_educations_candidate_id ON candidate_management.educations(candidate_id);
CREATE INDEX idx_educations_degree ON candidate_management.educations(degree);
CREATE INDEX idx_educations_institution ON candidate_management.educations(institution_name);

-- Experience indexes
CREATE INDEX idx_experiences_candidate_id ON candidate_management.experiences(candidate_id);
CREATE INDEX idx_experiences_company ON candidate_management.experiences(company_name);
CREATE INDEX idx_experiences_job_title ON candidate_management.experiences(job_title);
CREATE INDEX idx_experiences_is_current ON candidate_management.experiences(is_current);

-- Skills indexes
CREATE INDEX idx_candidate_skills_candidate_id ON candidate_management.candidate_skills(candidate_id);
CREATE INDEX idx_candidate_skills_skill_id ON candidate_management.candidate_skills(skill_id);

-- Job applications indexes
CREATE INDEX idx_job_applications_candidate_id ON candidate_management.job_applications(candidate_id);
CREATE INDEX idx_job_applications_job_id ON candidate_management.job_applications(job_id);
CREATE INDEX idx_job_applications_status ON candidate_management.job_applications(status);
CREATE INDEX idx_job_applications_application_date ON candidate_management.job_applications(application_date);

-- Job management indexes
CREATE INDEX idx_jobs_status ON job_management.jobs(status);
CREATE INDEX idx_jobs_job_code ON job_management.jobs(job_code);
CREATE INDEX idx_jobs_company_name ON job_management.jobs(company_name);
CREATE INDEX idx_skills_skill_name ON job_management.skills(skill_name);

-- Document Processing indexes
CREATE INDEX idx_documents_candidate_id ON document_processing.documents(candidate_id);
CREATE INDEX idx_documents_type ON document_processing.documents(document_type);
CREATE INDEX idx_documents_processing_status ON document_processing.documents(processing_status);
CREATE INDEX idx_documents_quality_score ON document_processing.documents(quality_score);
CREATE INDEX idx_documents_uploaded_at ON document_processing.documents(uploaded_at);

-- Processing queue indexes
CREATE INDEX idx_processing_queue_status ON document_processing.processing_queue(status);
CREATE INDEX idx_processing_queue_priority ON document_processing.processing_queue(priority DESC);
CREATE INDEX idx_processing_queue_scheduled_at ON document_processing.processing_queue(scheduled_at);
CREATE INDEX idx_processing_queue_assigned_to ON document_processing.processing_queue(assigned_to);

-- Verification tasks indexes
CREATE INDEX idx_verification_tasks_status ON document_processing.verification_tasks(status);
CREATE INDEX idx_verification_tasks_assigned_to ON document_processing.verification_tasks(assigned_to);
CREATE INDEX idx_verification_tasks_document_id ON document_processing.verification_tasks(document_id);

-- API usage logs indexes
CREATE INDEX idx_api_usage_logs_timestamp ON document_processing.api_usage_logs(timestamp);
CREATE INDEX idx_api_usage_logs_provider ON document_processing.api_usage_logs(api_provider);
CREATE INDEX idx_api_usage_logs_document_id ON document_processing.api_usage_logs(document_id);

-- Audit logs indexes
CREATE INDEX idx_audit_logs_user_id ON document_processing.audit_logs(user_id);
CREATE INDEX idx_audit_logs_timestamp ON document_processing.audit_logs(timestamp);
CREATE INDEX idx_audit_logs_action ON document_processing.audit_logs(action);

-- =============================================
-- CREATE VIEWS FOR COMMON QUERIES
-- =============================================

-- View for candidate overview with document counts
CREATE VIEW candidate_management.candidate_overview AS
SELECT 
    c.id,
    c.candidate_code,
    c.first_name,
    c.last_name,
    CONCAT(c.first_name, ' ', c.last_name) as full_name,
    c.email,
    c.status,
    c.document_quality_score,
    c.processing_stage,
    c.created_at,
    COUNT(d.id) as total_documents,
    COUNT(CASE WHEN d.processing_status = 'completed' THEN 1 END) as completed_documents,
    COUNT(CASE WHEN d.processing_status = 'review_required' THEN 1 END) as review_required_documents,
    AVG(d.quality_score) as avg_document_quality
FROM candidate_management.candidates c
LEFT JOIN document_processing.documents d ON c.id = d.candidate_id
GROUP BY c.id, c.candidate_code, c.first_name, c.last_name, c.email, c.status, c.document_quality_score, c.processing_stage, c.created_at;

-- View for verification queue
CREATE VIEW document_processing.verification_queue AS
SELECT 
    vt.id,
    vt.task_type,
    vt.status,
    vt.created_at,
    d.document_type,
    d.original_filename,
    c.candidate_code,
    CONCAT(c.first_name, ' ', c.last_name) as candidate_name,
    u.full_name as assigned_to_name,
    vt.time_spent_minutes
FROM document_processing.verification_tasks vt
JOIN document_processing.documents d ON vt.document_id = d.id
JOIN candidate_management.candidates c ON d.candidate_id = c.id
LEFT JOIN document_processing.users u ON vt.assigned_to = u.id
ORDER BY vt.created_at DESC;

-- View for candidate complete profile
CREATE VIEW candidate_management.candidate_complete_profile AS
SELECT 
    c.*,
    COUNT(e.id) as education_count,
    COUNT(exp.id) as experience_count,
    COUNT(cs.id) as skills_count,
    COUNT(ja.id) as applications_count
FROM candidate_management.candidates c
LEFT JOIN candidate_management.educations e ON c.id = e.candidate_id
LEFT JOIN candidate_management.experiences exp ON c.id = exp.candidate_id
LEFT JOIN candidate_management.candidate_skills cs ON c.id = cs.candidate_id
LEFT JOIN candidate_management.job_applications ja ON c.id = ja.candidate_id
GROUP BY c.id;

-- =============================================
-- INSERT DEFAULT DATA
-- =============================================

-- Insert default admin user
INSERT INTO document_processing.users (email, password_hash, full_name, role) VALUES 
('admin@imp.com', crypt('admin123', gen_salt('bf')), 'System Administrator', 'admin');

-- Insert default system settings
INSERT INTO document_processing.system_settings (setting_key, setting_value, description) VALUES 
('quality_threshold_auto', '75', 'Quality score threshold for automatic processing'),
('quality_threshold_review', '50', 'Quality score threshold below which manual review is required'),
('max_file_size_mb', '50', 'Maximum file size allowed for upload in MB'),
('supported_file_types', '["pdf", "doc", "docx", "jpg", "jpeg", "png"]', 'Supported file types for upload'),
('api_cost_limit_monthly', '500', 'Monthly API cost limit in USD'),
('default_processing_timeout', '300', 'Default processing timeout in seconds');

-- Insert some default skills
INSERT INTO job_management.skills (skill_name, category, description) VALUES 
('Microsoft Office', 'Software', 'Proficiency in Microsoft Office Suite'),
('English Communication', 'Language', 'Written and verbal English communication'),
('Arabic Language', 'Language', 'Arabic language proficiency'),
('Project Management', 'Management', 'Project planning and execution'),
('Customer Service', 'Soft Skills', 'Customer interaction and support'),
('Data Entry', 'Technical', 'Accurate data input and management'),
('Accounting', 'Finance', 'Financial record keeping and analysis'),
('Welding', 'Technical', 'Metal welding and fabrication'),
('Construction', 'Technical', 'Building and construction work'),
('Nursing', 'Healthcare', 'Patient care and medical assistance');

-- Grant permissions to schemas
GRANT ALL PRIVILEGES ON SCHEMA n8n TO imp_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA n8n TO imp_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA n8n TO imp_admin;

GRANT ALL PRIVILEGES ON SCHEMA candidate_management TO imp_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA candidate_management TO imp_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA candidate_management TO imp_admin;

GRANT ALL PRIVILEGES ON SCHEMA job_management TO imp_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA job_management TO imp_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA job_management TO imp_admin;

GRANT ALL PRIVILEGES ON SCHEMA document_processing TO imp_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA document_processing TO imp_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA document_processing TO imp_admin;

-- Create a trigger to update the updated_at column
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply the trigger to relevant tables
CREATE TRIGGER update_candidates_updated_at BEFORE UPDATE ON candidate_management.candidates FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_job_applications_updated_at BEFORE UPDATE ON candidate_management.job_applications FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_jobs_updated_at BEFORE UPDATE ON job_management.jobs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON document_processing.users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_system_settings_updated_at BEFORE UPDATE ON document_processing.system_settings FOR EACH ROW EXECUTE FUNCTION update_updated_at_column(); 