Computation Models
==================

TRUST implements data-driven controller synthesis using sum-of-squares (SOS)
optimisation. This section describes the theoretical foundation and the two
synthesis modes: **Stability** (CLF) and **Safety** (CBC).

Data-Driven Approach
--------------------

TRUST uses Willems *et al.*'s fundamental lemma to avoid explicit system
identification. Instead of learning a model :math:`\dot{x} = f(x,u)` or
:math:`x_{k+1} = f(x_k, u_k)`, it works directly with trajectory data:

- :math:`X_0 \in \mathbb{R}^{n \times T}` --- state samples at times :math:`t_0, \ldots, t_{T-1}`
- :math:`X_1 \in \mathbb{R}^{n \times T}` --- state samples at times :math:`t_1, \ldots, t_T`
- :math:`U_0 \in \mathbb{R}^{m \times T}` --- input samples at times :math:`t_0, \ldots, t_{T-1}`

A **rank condition** ensures the data is persistently exciting:

- For linear systems: :math:`T > n` and :math:`\text{rank}(X_0) = n`
- For polynomial systems: :math:`T > N` and :math:`\text{rank}(N_0) = N`, where
  :math:`N` is the number of monomial terms and :math:`N_0` evaluates the
  monomials at each sample

----

Computation Workflow
--------------------

All four system classes follow the same high-level workflow:

.. code-block:: text

   1. Parse input data (X0, X1, U0 matrices)
   2. Check rank conditions for persistent excitation
   3. Formulate SOS/SDP optimisation problem
   4. Solve via MOSEK through PICOS / SumOfSquares
   5. Extract H, Z matrices; compute P = Z^{-1}
   6. Form Lyapunov/barrier function and controller
   7. Validate SOS decompositions
   8. Return results (function, controller, level sets)

For **linear systems**, the optimisation is a semidefinite program (SDP)
formulated directly with PICOS. For **nonlinear polynomial systems**, the
optimisation uses SOS constraints via the SumOfSquares library.

----

Key Concepts
------------

Monomial Vector :math:`M(x)`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For nonlinear polynomial systems, the user provides a monomial basis
:math:`M(x) = [m_1(x), m_2(x), \ldots, m_N(x)]^T`. For example, for a 2D
system with quadratic interactions::

   M(x) = [x1, x2, x1*x2]

The :math:`N_0` matrix is formed by evaluating :math:`M(x)` at each data sample.

Theta Matrix :math:`\Theta(x)`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For discrete-time polynomial systems, the :math:`\Theta(x)` matrix (shape
:math:`N \times n`) is the Jacobian of the monomial vector:

.. math::

   \Theta(x)_{ij} = \frac{\partial m_i(x)}{\partial x_j}

TRUST can compute this automatically from the monomials via the **Autofill**
button.

Matrix :math:`P` and the Lyapunov/Barrier Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core output is a positive definite matrix :math:`P` such that:

- **Stability**: :math:`V(x) = x^T P x` (linear) or :math:`V(x) = M(x)^T P M(x)` (polynomial)
  is a Control Lyapunov Function
- **Safety**: :math:`B(x) = x^T P x` or :math:`B(x) = M(x)^T P M(x)` is a
  Control Barrier Certificate

The controller is given by :math:`u = U_0 H P x` (linear) or
:math:`u = U_0 H(x) P M(x)` (polynomial).

----

Classes
-------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Stability
      :link: stability
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Control Lyapunov Function synthesis for all four system classes.
      Guarantees asymptotic stability of the closed-loop system.

   .. grid-item-card:: Safety Barrier
      :link: safety_barrier
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Control Barrier Certificate synthesis for all four system classes.
      Guarantees trajectories remain in the safe set.


.. toctree::
   :hidden:

   stability
   safety_barrier
