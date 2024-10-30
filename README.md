# SinTrajBC
Single trajectory control with barrier certificates

## Installation

Clone the project from Github:
```bash
git clone https://github.com/thatgardnerone/TRUST.git
# or use ssh: git clone git@github.com:thatgardnerone/TRUST.git
cd TRUST
```

Note: this project uses Python 3.12.2. If you don't have it installed, you can download it from the [official website](https://www.python.org/downloads/).

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

And finally, in a new terminal (remember to reactivate your venv with the source command above) run the flask development server:
```bash
FLASK_APP=main.py FLASK_ENV=development FLASK_DEBUG=1 flask run
```
