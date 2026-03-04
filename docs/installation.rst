Installation
============

TRUST can be used in three ways: via the hosted server, installed locally with
Docker, or set up for local development with Poetry.

Hosted Server
-------------

A hosted version of TRUST is available at https://trust.tgo.dev for immediate
use. No installation required.

.. warning::

   The hosted server has limited capacity and may not perform as well as a local
   installation for large problems or concurrent users.

----

Docker Installation (Recommended)
----------------------------------

Docker provides the simplest installation experience with all dependencies
pre-configured.

Prerequisites
~~~~~~~~~~~~~

**Docker**

Install Docker by following the instructions for your operating system at
https://docs.docker.com/get-docker/.

**MOSEK License**

TRUST relies on the MOSEK solver for SOS optimisation. A free trial license is
available (free for academic users). Request one at
https://www.mosek.com/license/request/?i=acp.

The license file will be emailed to you. You will upload it through the TRUST
GUI when running the tool.

Step 1: Download TRUST
~~~~~~~~~~~~~~~~~~~~~~

Download the repository from https://github.com/Kiguli/TRUST. Click the green
"Code" button, then "Download ZIP". Unzip the folder and rename it to ``TRUST``.

Alternatively, clone with Git:

.. code-block:: bash

   git clone https://github.com/Kiguli/TRUST.git
   cd TRUST

Step 2: Copy the environment file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::

   .. tab-item:: macOS / Linux

      .. code-block:: bash

         cp .env.example .env

   .. tab-item:: Windows

      .. code-block:: doscon

         copy .env.example .env

Step 3: Build and run
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker compose up --build -d

Step 4: Access TRUST
~~~~~~~~~~~~~~~~~~~~

Open your web browser and navigate to:

   http://127.0.0.1:8000

.. tip::

   To stop the container, run ``docker compose down``. To restart it later,
   run ``docker compose up -d`` (no ``--build`` needed unless code has changed).

----

Local Development (without Docker)
-----------------------------------

For development or server deployment without Docker, set up TRUST using pyenv
and Poetry.

Install pyenv
~~~~~~~~~~~~~

pyenv allows you to manage multiple Python versions.

.. tab-set::

   .. tab-item:: macOS (Homebrew)

      .. code-block:: bash

         brew install pyenv
         echo 'eval "$(pyenv init -)"' >> ~/.zshrc
         source ~/.zshrc

   .. tab-item:: Linux

      .. code-block:: bash

         curl https://pyenv.run | bash
         echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
         echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
         echo 'eval "$(pyenv init -)"' >> ~/.bashrc
         source ~/.bashrc

Install Python 3.12.10
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pyenv install 3.12.10

Install Poetry
~~~~~~~~~~~~~~

.. code-block:: bash

   pip install poetry

Set up the project
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd TRUST

   # Use Python 3.12.10 for this project
   pyenv local 3.12.10

   # Install dependencies
   poetry install

   # Copy environment file
   cp .env.example .env

Build the frontend
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd vite
   npm ci
   npm run build
   cd ..

Run the application
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   poetry run python main.py

The application will be available at ``https://127.0.0.1:5000``.

.. note::

   The local development server uses ad-hoc SSL. Your browser may show a
   security warning --- this is expected and can be safely bypassed for local
   development.

----

Running Tests
-------------

.. code-block:: bash

   poetry run pytest

Dependency Auditing
-------------------

.. code-block:: bash

   poetry run pip-audit     # Python vulnerabilities
   cd vite && npm audit     # npm vulnerabilities
