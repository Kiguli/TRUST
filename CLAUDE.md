# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TRUST is a Python Flask web application for data-driven controller synthesis of dynamical systems with unknown mathematical models. It synthesizes Control Lyapunov Functions (CLF) or Control Barrier Certificates (CBC) using sum-of-squares (SOS) optimization based solely on input-state trajectory data.

Supports four system classes:
- Continuous-time nonlinear polynomial systems (ct-NPS)
- Continuous-time linear systems (ct-LS)
- Discrete-time nonlinear polynomial systems (dt-NPS)
- Discrete-time linear systems (dt-LS)

## Development Commands

### Docker (recommended)
```bash
cp .env.example .env
docker compose up --build -d
# Access at http://127.0.0.1:8000
```

### Local Development (Poetry)
```bash
# Ensure Python 3.12.10 via pyenv
pyenv install 3.12.10
pyenv local 3.12.10

# Install dependencies
poetry install

# Copy environment file
cp .env.example .env

# Build frontend
cd vite && npm ci && npm run build && cd ..

# Run application
poetry run python main.py  # Runs on port 5000 with SSL
```

### Running Tests
```bash
poetry run pytest
```

### Dependency Auditing
```bash
poetry run pip-audit  # Python vulnerabilities
cd vite && npm audit  # npm vulnerabilities
```

## Architecture

### Backend (Python/Flask)
- `main.py` - Application entry point, creates Flask app
- `app/__init__.py` - Flask app factory with Inertia.js and Vite integration
- `app/http/controllers/dashboard_controller.py` - Main route handler, processes form submissions and file uploads

### Computation Models (`app/models/`)
- `barrier.py` - Base Barrier class with shared properties (state space, monomials, data parsing)
- `safety_barrier.py` - SafetyBarrier class implementing CBC synthesis for all 4 system types
- `stability.py` - Stability class implementing CLF synthesis for all 4 system types

Key computation flow:
1. Parse input data (X0, X1, U0 matrices) from form or CSV files
2. Check rank conditions for persistent excitation
3. Solve SOS optimization using MOSEK solver via PICOS/SumOfSquares
4. Return Lyapunov/barrier function and controller expressions

### Frontend (Vue 3/Inertia.js)
- `vite/` - Vue 3 SPA with Tailwind CSS
- `vite/main.js` - Inertia.js app setup
- `vite/js/Components/Pages/Dashboard.vue` - Main interface
- Uses Atomic Design: Atoms → Molecules → Organisms → Pages

### Data Storage
- `storage/cases/` - Example benchmark datasets with CSV files (X0T, X1T, U0T matrices)
- Uploaded datasets are parsed in memory and never written to disk. An uploaded MOSEK licence only exists in a private temporary file while its own calculation runs, and is deleted afterwards.

## Security Notes

- Never pass user input to `sympy.sympify`, `parse_expr`, `S()` or `eval`: they execute code. Parse monomials and Theta(x) entries with `parse_polynomial` / `parse_monomials` from `app/models/expressions.py`, which only accept polynomials in `x1`…`xn` and bound their size.
- Never build filesystem paths from client-supplied filenames. Parse uploads in memory; the filename is only used to choose the format.
- Run every solve for a request inside `_mosek_license` in the dashboard controller. PICOS shares one MOSEK environment per process, and MOSEK fixes an environment's licence when it is created, so each calculation gets a new environment licensed by its own upload (or the server's default licence) and discarded afterwards.

## Key Dependencies

- **MOSEK**: Commercial SOS solver (requires license file uploaded via GUI)
- **PICOS/SumOfSquares**: Python SOS optimization interfaces
- **SymPy**: Symbolic mathematics for polynomial manipulation
- **NumPy**: Matrix operations and linear algebra
- **Flask-Inertia**: Server-side rendering bridge to Vue.js frontend

## Environment Variables

Key variables in `.env`:
- `FLASK_DEBUG` / `FLASK_ENV` - Development mode settings
- `FLASK_KEY` - Secret key for sessions
- `SENTRY_DSN` - Error monitoring (optional)
- `GUNICORN_PORT` - Docker container port (default: 8000)
- `FLASK_RUN_HOST` - Interface the development server (`main.py`) listens on (default: 127.0.0.1)
- `MAX_CONTENT_LENGTH` - Largest accepted request body in bytes (default: 16 MiB)
