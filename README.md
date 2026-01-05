# Another Way of Seeing — Math Grapher + Arduino Plotter (Flask)

A Flask-based math graphing project that parses **SymPy-compatible expressions** (e.g., `sin(x)`, `x**2`) and generates:
- **On-screen plot data** for a web UI
- **Discrete grid/cell paths** suitable for **Arduino/stepper-style drawing**
- Optional **AI-based “next expression” recommendations** 

This project is built around the idea of **“Another Way of Seeing”** — translating mathematical functions into a form that can be *visually explored* and even *physically rendered* through motion control.

---
## Security / Redacted Code Notice

This repository contains a public-facing version of the project.  
Some key codes (e.g., API keys, credentials) and certain core logic sections have been intentionally hidden or removed because this application is currently being used in collaboration with a third-party company.

If you need access for legitimate review, deployment, or partnership purposes, please contact the maintainer.

## Key Features

- **Expression parsing (SymPy)**  
  Parses user input safely (expressions in `x` only; no assignments).

- **Plot data API**
  - `GET /data` returns sample points for browser plotting.

- **Coordinate/path generation**
  - `GET /coords` returns float points and/or stepper-ready integer cell paths.
  - `GET /path` provides a legacy path mode (float or integer cell output).

- **Centered grid mapping (N×N)**
  Generates paths on a centered integer grid (default `GRID_N=91`) with Bresenham line densification.

- **Arduino serial sending**
  Can send generated cell paths to a microcontroller via `pyserial`.

- **Recommendations**
  - `POST /recommend` suggests 3 next expressions to try.
  - Includes a local fallback if OpenAI is unavailable.

- **Auth system (SQLite)**
  Basic registration/login/logout using hashed passwords.

---

## Tech Stack

- **Backend:** Python + Flask  
- **Math Engine:** SymPy + NumPy  
- **Auth:** SQLite + Werkzeug password hashing  
- **Optional Hardware:** Arduino + pyserial  
- **Optional AI:** OpenAI Chat Completions

---

## Project Structure

- `app.py`
- `templates/`
  - `landing.html`
  - `index.html`
  - `login.html`
  - `register.html`
- `auth.db` 

---

## Setup

### 1) Install dependencies
```bash
pip install flask numpy sympy werkzeug
