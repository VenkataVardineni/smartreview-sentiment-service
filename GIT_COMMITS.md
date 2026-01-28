# Git Commit History

This document outlines the commit structure for the SmartReview Sentiment Service project.

## Commit Structure

The project follows a structured commit history as requested:

### 1. `chore: scaffold project structure`
**Files:**
- `src/__init__.py`
- `src/data/__init__.py`
- `src/models/__init__.py`
- `src/training/__init__.py`
- `src/api/__init__.py`

**Purpose:** Initial project scaffolding with directory structure and Python package initialization files.

### 2. `feat: add dataset preparation script`
**Files:**
- `src/data/prepare_dataset.py`

**Purpose:** Implements dataset preparation pipeline that downloads IMDB reviews, cleans text, and splits into train/val/test sets.

### 3. `feat: add PyTorch model and basic training pipeline`
**Files:**
- `src/models/text_classifier.py` - LSTM-based text classifier model
- `src/training/train.py` - Training script with metrics logging

**Purpose:** Implements the PyTorch sentiment classification model and complete training pipeline with CSV metrics logging.

### 4. `feat: add FastAPI inference API and Docker setup`
**Files:**
- `src/api/main.py` - FastAPI application with /predict endpoint
- `Dockerfile` - Container definition for API
- `docker-compose.yml` - Docker orchestration
- `requirements.txt` - Python dependencies

**Purpose:** Implements the REST API for sentiment inference and containerization setup for deployment.

### Additional Commits

- `docs: add comprehensive README and setup instructions` - Complete documentation
- `feat: add React frontend with modern UI` - React frontend with Framer Motion

## Viewing Commits

```bash
# View commit history
git log --oneline

# View detailed commit
git show <commit-hash>

# View changes in a commit
git diff <commit-hash>^..<commit-hash>
```

## Pushing to GitHub

```bash
# Check remote
git remote -v

# Push to GitHub
git push origin main

# Or set new remote
git remote set-url origin <your-repo-url>
git push -u origin main
```

