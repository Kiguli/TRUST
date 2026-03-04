Usage Guide
===========

This guide walks through using the TRUST web application to synthesise
controllers for dynamical systems with unknown models.

Overview
--------

TRUST provides a single-page interface with three columns:

1. **Input Options** (left) --- configure the problem type and provide trajectory data
2. **Additional Inputs** (centre) --- set monomials, :math:`\Theta(x)`, and region bounds
3. **Results** (right) --- view the synthesised function, controller, and diagnostics

----

Step 1: Upload MOSEK License
-----------------------------

Before running any computation, upload your MOSEK license file using the
file picker at the top of the input panel. The license is required for the
SOS solver and must be uploaded each session.

.. tip::

   Request a free MOSEK license at https://www.mosek.com/license/request/?i=acp
   (free for academic users).

----

Step 2: Select Problem Type
-----------------------------

Configure three settings using the radio selectors:

**Class**
   - *Discrete-Time* --- for systems of the form :math:`x_{k+1} = f(x_k, u_k)`
   - *Continuous-Time* --- for systems of the form :math:`\dot{x} = f(x, u)`

**Model**
   - *Linear* --- for linear systems (no monomials required)
   - *Non-Linear Polynomial* --- for polynomial systems (requires monomials and
     possibly :math:`\Theta(x)`)

**Specification**
   - *Stability* --- synthesise a Control Lyapunov Function (CLF) and stabilising controller
   - *Safety* --- synthesise a Control Barrier Certificate (CBC) and safety controller

----

Step 3: Provide Trajectory Data
---------------------------------

TRUST requires three data matrices from a single input-state trajectory:

- **X0** --- state measurements at times :math:`t_0, t_1, \ldots, t_{T-1}` (shape: :math:`n \times T`)
- **X1** --- state measurements at times :math:`t_1, t_2, \ldots, t_T` (shape: :math:`n \times T`)
- **U0** --- input measurements at times :math:`t_0, t_1, \ldots, t_{T-1}` (shape: :math:`m \times T`)

where :math:`n` is the state dimension, :math:`m` is the input dimension, and
:math:`T` is the number of samples.

Each dataset can be provided in two ways:

**Manual entry**
   Type or paste the matrix values directly. Rows are separated by newlines,
   columns by commas.

**File upload**
   Upload a CSV file. Drag-and-drop is supported. The files in
   ``storage/cases/`` use this format (e.g., ``X0T.csv``, ``X1T.csv``, ``U0T.csv``).

.. note::

   The data must satisfy a rank condition (persistent excitation). For linear
   systems: :math:`T > n` and :math:`X_0` must be full row-rank. For nonlinear
   polynomial systems: :math:`T > N` and the :math:`N_0` matrix must be full
   row-rank, where :math:`N` is the number of monomial terms.

----

Step 4: Configure Additional Inputs
-------------------------------------

Depending on the problem type, additional inputs may be required.

Monomials (Non-Linear Polynomial only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enter the monomial basis terms separated by semicolons. Terms must use symbols
``x1``, ``x2``, ..., ``xn`` matching the state dimension.

Example for a 2D system::

   x1; x2; x1*x2

Example for a 3D system::

   x1; x2; x3; x1*x2; x2*x3; x1*x3

Theta(x) Matrix (Discrete-Time NPS only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :math:`\Theta(x)` matrix relates the Jacobian of the monomial vector
:math:`M(x)` to the state variables. It has shape :math:`N \times n`.

Click the **Autofill** button to compute :math:`\Theta(x)` automatically from
the specified monomials.

State Space, Initial Set, and Unsafe Sets (Safety only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For safety specifications, define the regions as axis-aligned boxes by
providing lower and upper bounds for each state dimension:

- **State Space** --- the overall operating region
- **Initial Set** --- the set of initial conditions (shown in blue in plots)
- **Unsafe Set(s)** --- one or more unsafe regions to avoid (shown in red)

----

Step 5: Run the Computation
-----------------------------

Click the **Submit** button or press :kbd:`Cmd+Enter` (macOS) /
:kbd:`Ctrl+Enter` (Windows/Linux) to start the computation.

----

Interpreting Results
--------------------

On success, the results panel displays:

**Lyapunov / Barrier Function**
   The synthesised function :math:`V(x)` (for stability) or :math:`B(x)` (for safety),
   expressed as :math:`x^T P x` (linear) or :math:`M(x)^T P M(x)` (polynomial),
   along with the computed matrix :math:`P`.

**Controller**
   The state-feedback controller expression :math:`u = U_0 H P x` (linear) or
   :math:`u = U_0 H(x) P M(x)` (polynomial), along with the computed matrix :math:`H`.

**Level Sets** (Safety only)
   The scalar values :math:`\gamma` and :math:`\lambda` defining the safe
   operating region: :math:`\{x : B(x) \leq \gamma\}` is contained in the safe
   set, and :math:`\{x : B(x) \leq \lambda\}` contains the initial set.

**Timing and Memory**
   Wall-clock time and peak memory usage for the computation.

----

Error Handling
--------------

TRUST provides descriptive error messages for common issues:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Error Message
     - Cause
   * - "Provided spaces are not valid..."
     - Invalid state space, initial set, or unsafe set bounds
   * - "Theta_x should be of shape (N, n)"
     - :math:`\Theta(x)` matrix has incorrect dimensions
   * - "Monomial terms should be split by semicolon"
     - Commas used instead of semicolons in monomials
   * - "Monomials must be in terms of x1 (to xn)"
     - Monomial symbols don't match state dimension
   * - "The number of samples, T, must be greater than..."
     - Rank condition not met (insufficient data)
   * - "The X0/N0 data is not full row-rank"
     - Data does not satisfy persistent excitation
   * - "Unable to parse uploaded file(s)"
     - Invalid file format
   * - "Solution Failure"
     - MOSEK solver could not find a feasible solution
   * - "No SOS decomposition found"
     - Solution exists but lacks valid SOS decomposition
   * - "Constraints are not sum-of-squares"
     - Solution constraints are not valid SOS
