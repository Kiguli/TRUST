Safety Barrier (CBC Synthesis)
==============================

The ``SafetyBarrier`` class synthesises a Control Barrier Certificate (CBC)
:math:`B(x)` and a safety controller :math:`u(x)` for dynamical systems with
unknown models, using only trajectory data.

Overview
--------

A CBC :math:`B(x)` guarantees that trajectories starting from an initial set
remain within a safe region by satisfying:

1. :math:`B(x) \leq \gamma` for all :math:`x` in the initial set
2. :math:`B(x) > \lambda` for all :math:`x` in the unsafe set(s)
3. :math:`B(x)` does not increase along trajectories within the safe operating region

where :math:`0 < \gamma < \lambda` are computed level-set scalars.

TRUST finds :math:`B(x) = x^T P x` (linear systems) or
:math:`B(x) = M(x)^T P M(x)` (polynomial systems), where :math:`P \succ 0` is
computed via SOS optimisation.

----

Linear Systems
--------------

Continuous-Time (ct-LS)
~~~~~~~~~~~~~~~~~~~~~~~

For :math:`\dot{x} = Ax + Bu` with unknown :math:`A, B`, TRUST solves:

.. math::

   \text{find } H, Z \text{ such that:} \\
   Z = X_0 H, \quad Z - \varepsilon I \succeq 0, \quad
   H^T X_1^T + X_1 H \preceq 0

Then adds SOS constraints for the level sets :math:`\gamma, \lambda` using
Lagrangian multipliers over the state space, initial set, and unsafe set(s).

Discrete-Time (dt-LS)
~~~~~~~~~~~~~~~~~~~~~

For :math:`x_{k+1} = Ax_k + Bu_k`, TRUST solves:

.. math::

   Z = X_0 H, \quad Z - \varepsilon I \succeq 0, \quad
   \begin{bmatrix} Z & X_1 H \\ H^T X_1^T & Z \end{bmatrix} \succeq 0

followed by level-set constraints.

----

Nonlinear Polynomial Systems
-----------------------------

Continuous-Time (ct-NPS)
~~~~~~~~~~~~~~~~~~~~~~~~

For polynomial systems :math:`\dot{x} = f(x) + g(x)u`:

.. math::

   N_0 H(x) = Z, \quad Z - \varepsilon I \succeq 0

with Lie derivative constraint:

.. math::

   \frac{\partial M}{\partial x} X_1 H(x) + H(x)^T X_1^T \frac{\partial M}{\partial x}^T \preceq 0

The barrier function is :math:`B(x) = M(x)^T P M(x)` with :math:`P = Z^{-1}`.

Discrete-Time (dt-NPS)
~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   N_0 H(x) = \Theta(x) Z, \quad Z - \varepsilon I \succeq 0

with Schur complement SOS constraint, followed by level-set constraints.

----

Level Sets
----------

After finding :math:`P` and :math:`H(x)`, TRUST solves a second SOS problem to
find the level-set scalars :math:`\gamma` and :math:`\lambda`:

.. math::

   0 < \gamma < \lambda

These define the safe operating regions:

- :math:`\{x : B(x) \leq \gamma\}` --- contains the initial set (shown as dashed blue)
- :math:`\{x : B(x) \leq \lambda\}` --- is contained within the safe set (shown as dashed red)
- The unsafe set(s) are guaranteed to satisfy :math:`B(x) > \lambda`

The constraints are enforced using SOS Lagrangian multipliers over the
user-defined state space, initial set, and unsafe set(s).

----

Input Format
------------

The ``SafetyBarrier`` class (via its ``Barrier`` base class) accepts a dictionary
with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Key
     - Type
     - Description
   * - ``mode``
     - str
     - Must be ``"Safety"``
   * - ``model``
     - str
     - ``"Linear"`` or ``"Non-Linear Polynomial"``
   * - ``timing``
     - str
     - ``"Discrete-Time"`` or ``"Continuous-Time"``
   * - ``X0``
     - array/str
     - State data matrix :math:`X_0` (:math:`n \times T`)
   * - ``X1``
     - array/str
     - State data matrix :math:`X_1` (:math:`n \times T`)
   * - ``U0``
     - array/str
     - Input data matrix :math:`U_0` (:math:`m \times T`)
   * - ``monomials``
     - str (JSON)
     - Monomial terms (NPS only)
   * - ``theta_x``
     - str (JSON)
     - :math:`\Theta(x)` matrix (dt-NPS only)
   * - ``stateSpace``
     - str (JSON)
     - Bounds for each state dimension
   * - ``initialState``
     - str (JSON)
     - Bounds for the initial set
   * - ``unsafeStates``
     - str (JSON)
     - List of bounds for unsafe set(s)

----

Output Format
-------------

The ``calculate()`` method returns a dictionary:

.. code-block:: python

   {
       "function": {
           "expression": {"x<sup>T</sup>Px": "[[...]]"},
           "values": {"P": "[[...]]"}
       },
       "controller": {
           "expression": {"U<sub>0</sub>HPx": "[[...]]"},
           "values": {"H": "[[...]]"}
       },
       "gamma": "0.123",
       "lambda": "0.456"
   }

On error, the dictionary contains ``"error"`` and ``"description"`` keys instead.

----

API Reference
-------------

.. autoclass:: app.models.safety_barrier.SafetyBarrier
   :members: calculate, generate_polynomial
   :show-inheritance:

.. autoclass:: app.models.barrier.Barrier
   :members: calculate, generate_polynomial, parse_dataset
   :show-inheritance:
