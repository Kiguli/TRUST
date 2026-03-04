Benchmark Cases
===============

TRUST includes 17 benchmark case studies with pre-configured trajectory data.
Each case is stored in ``storage/cases/`` and can be used to verify the tool
or as a starting point for new analyses.

Case Overview
-------------

.. list-table::
   :header-rows: 1
   :widths: 35 10 10 10 15

   * - Case
     - Time
     - Model
     - Dim
     - Specifications
   * - DC Motor
     - ct
     - LS
     - 2
     - Stability, Safety
   * - Room Temperature System 1
     - ct
     - LS
     - 2
     - Stability, Safety
   * - Two-Tank System
     - ct
     - LS
     - 2
     - Stability, Safety
   * - High-Order System (4D)
     - ct
     - LS
     - 4
     - Stability, Safety
   * - High-Order System (6D)
     - ct
     - LS
     - 6
     - Stability, Safety
   * - High-Order System (8D)
     - ct
     - LS
     - 8
     - Stability, Safety
   * - Lotka-Volterra Predator-Prey
     - ct
     - NPS
     - 2
     - Stability, Safety
   * - Van der Pol Oscillator
     - ct
     - NPS
     - 2
     - Stability, Safety
   * - DC Motor
     - dt
     - LS
     - 2
     - Stability, Safety
   * - Room Temperature System 1
     - dt
     - LS
     - 2
     - Stability, Safety
   * - Room Temperature System 2
     - dt
     - LS
     - 3
     - Stability, Safety
   * - Two-Tank System
     - dt
     - LS
     - 2
     - Stability, Safety
   * - High-Order System (4D)
     - dt
     - LS
     - 4
     - Stability, Safety
   * - High-Order System (6D)
     - dt
     - LS
     - 6
     - Stability, Safety
   * - High-Order System (8D)
     - dt
     - LS
     - 8
     - Stability, Safety
   * - Lotka-Volterra Predator-Prey
     - dt
     - NPS
     - 2
     - Stability, Safety
   * - Lorenz Attractor
     - dt
     - NPS
     - 3
     - Stability, Safety

----

File Format
-----------

Each case directory contains up to six CSV data files and one supplementary
input file:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - ``stability_X0T.csv``
     - :math:`X_0` matrix for stability analysis
   * - ``stability_X1T.csv``
     - :math:`X_1` matrix for stability analysis
   * - ``stability_U0T.csv``
     - :math:`U_0` matrix for stability analysis
   * - ``safety_X0T.csv``
     - :math:`X_0` matrix for safety analysis
   * - ``safety_X1T.csv``
     - :math:`X_1` matrix for safety analysis
   * - ``safety_U0T.csv``
     - :math:`U_0` matrix for safety analysis
   * - ``supplementary_inputs.txt``
     - Monomials, :math:`\Theta(x)`, state space, initial/unsafe sets

CSV files contain comma-separated floating-point values, one row per state
dimension.

Supplementary Inputs
~~~~~~~~~~~~~~~~~~~~

The ``supplementary_inputs.txt`` file lists additional inputs required for each
specification, organised under ``-- Safety --`` and ``-- Stability --`` headers.

Example (from ``ctNPS_lotka_volterra_predator_prey_model``)::

   -- Safety --
   Monomials: x1; x2; x1*x2
   State Space:
     x1: 0, 6
     x2: 0, 6
   Initial State:
     x1: 1.5, 2
     x2: 0.75, 1
   Unsafe State:
     x1: 4, 5
     x2: 3, 4

----

Using a Benchmark Case
-----------------------

1. Select the appropriate **Class** and **Model** matching the case prefix
   (e.g., ``ctNPS`` = Continuous-Time, Non-Linear Polynomial)
2. Upload the corresponding CSV files (``X0T.csv``, ``X1T.csv``, ``U0T.csv``)
   for your chosen specification (stability or safety)
3. Enter the supplementary inputs from ``supplementary_inputs.txt``:
   monomials, :math:`\Theta(x)` (or use Autofill), and region bounds
4. Click **Submit** to run the computation

----

Example Results
---------------

The figures below illustrate safety results for selected benchmark cases. In
each plot:

- Blue region: initial set
- Red region(s): unsafe set(s)
- Dashed blue line: :math:`\lambda` level set
- Dashed red line: :math:`\gamma` level set
- Black curves: closed-loop trajectories

.. note::

   The benchmark figures were rendered using external MATLAB scripts not
   included in the repository.
