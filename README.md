# SinTrajBC
Single trajectory control with barrier certificates

## Installation

It's recommended to use a virtual environment to install the dependencies. Use your favorite tool, or simply run:
```bash
python -m venv venv
source venv/bin/activate
```

Then get started by installing the python dependencies:
```bash
pip install -r requirements.txt
```

Make sure your local environment has all the relevant info:
```bash
cp .env.example .env
# Make sure you change any values you need, e.g. a different port
```

Next, in a separate terminal, install and run the front-end:
```bash
cd vite
npm install
npm run dev
```

And finally, run the flask development server:
```bash
FLASK_APP=main.py FLASK_ENV=development FLASK_DEBUG=1 flask run
```
