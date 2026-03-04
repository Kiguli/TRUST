Stability (CLF Synthesis)
=========================

The ``Stability`` class synthesises a Control Lyapunov Function (CLF)
:math:`V(x)` and a stabilising controller :math:`u(x)` for dynamical systems
with unknown models, using only trajectory data.

Overview
--------

A CLF :math:`V(x)` guarantees asymptotic stability of the closed-loop system by
satisfying:

1. :math:`V(x) > 0` for all :math:`x \neq 0` (positive definiteness)
2. :math:`V(x)` decreases along system trajectories under the synthesised controller

TRUST finds :math:`V(x) = x^T P x` (linear systems) or
:math:`V(x) = M(x)^T P M(x)` (polynomial systems), where :math:`P \succ 0` is
computed via SOS optimisation.

----

Linear Systems
--------------

Continuous-Time (ct-LS)
~~~~~~~~~~~~~~~~~~~~~~~

For :math:`\dot{x} = Ax + Bu` with unknown :math:`A, B`, TRUST solves the SDP:

.. math::

   \text{find } H, Z \text{ such that:} \\
   Z = X_0 H, \quad Z - \varepsilon I \succeq 0, \quad
   H^T X_1^T + X_1 H \preceq 0

The Lyapunov function is :math:`V(x) = x^T P x` with :math:`P = Z^{-1}`, and
the controller is :math:`u = U_0 H P x`.

Discrete-Time (dt-LS)
~~~~~~~~~~~~~~~~~~~~~

For :math:`x_{k+1} = Ax_k + Bu_k` with unknown :math:`A, B`, TRUST solves:

.. math::

   \text{find } H, Z \text{ such that:} \\
   Z = X_0 H, \quad Z - \varepsilon I \succeq 0, \quad
   \begin{bmatrix} Z & H^T X_1^T \\ X_1 H & Z \end{bmatrix} \succeq 0

The Schur complement condition ensures :math:`V(x_{k+1}) - V(x_k) < 0`.

----

Nonlinear Polynomial Systems
-----------------------------

Continuous-Time (ct-NPS)
~~~~~~~~~~~~~~~~~~~~~~~~

For polynomial systems :math:`\dot{x} = f(x) + g(x)u`, TRUST solves for
polynomial matrices :math:`H(x)` and constant matrix :math:`Z`:

.. math::

   N_0 H(x) = Z, \quad Z - \varepsilon I \succeq 0

with the Lie derivative constraint:

.. math::

   \frac{\partial M}{\partial x} X_1 H(x) + H(x)^T X_1^T \frac{\partial M}{\partial x}^T \preceq 0

The Lyapunov function is :math:`V(x) = M(x)^T P M(x)` with :math:`P = Z^{-1}`.

Discrete-Time (dt-NPS)
~~~~~~~~~~~~~~~~~~~~~~~

For polynomial systems :math:`x_{k+1} = f(x_k) + g(x_k)u_k`, TRUST solves:

.. math::

   N_0 H(x) = \Theta(x) Z, \quad Z - \varepsilon I \succeq 0

with a Schur complement condition analogous to the linear discrete-time case.

The user must provide the monomial vector :math:`M(x)` and the
:math:`\Theta(x)` matrix (or use the Autofill feature).

----

Input Format
------------

The ``Stability`` class accepts a dictionary with the following keys:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Key
     - Type
     - Description
   * - ``mode``
     - str
     - Must be ``"Stability"``
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
       }
   }

On error, the dictionary contains ``"error"`` and ``"description"`` keys instead.

----

API Reference
-------------

.. autoclass:: app.models.stability.Stability
   :members: calculate, parse_dataset, generate_polynomial
   :show-inheritance:
