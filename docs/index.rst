:html_theme.sidebar_secondary.remove:

.. meta::
   :description: TRUST — Data-driven controller synthesis for unknown dynamical systems
   :keywords: control synthesis, Lyapunov, barrier certificate, SOS optimization, data-driven

=====
TRUST
=====

**Data-driven controller synthesis for dynamical systems with unknown models**

TRUST synthesises Control Lyapunov Functions (CLF) or Control Barrier
Certificates (CBC), along with their corresponding controllers, using only a
single input-state trajectory from the unknown system. It implements
sum-of-squares (SOS) optimisation programs solely based on data to enforce
stability or safety properties.

.. grid:: 2 2 4 4
   :gutter: 3

   .. grid-item-card:: :octicon:`database;1.5em` Data-Driven
      :class-card: sd-border-0 sd-shadow-sm

      No mathematical model required. Uses only a single
      input-state trajectory satisfying a rank condition.

   .. grid-item-card:: :octicon:`verified;1.5em` SOS Optimisation
      :class-card: sd-border-0 sd-shadow-sm

      Rigorous sum-of-squares programs solved via MOSEK
      to guarantee stability or safety certificates.

   .. grid-item-card:: :octicon:`rows;1.5em` Four System Classes
      :class-card: sd-border-0 sd-shadow-sm

      Supports continuous-time and discrete-time systems,
      both linear and nonlinear polynomial.

   .. grid-item-card:: :octicon:`browser;1.5em` Web Interface
      :class-card: sd-border-0 sd-shadow-sm

      Intuitive reactive GUI built with Vue 3. Supports
      manual data entry and CSV file uploads.

----

Getting Started
===============

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Install TRUST via Docker or set up local development.

   .. grid-item-card:: Usage Guide
      :link: usage
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Learn how to use the web application: upload data,
      configure options, and interpret results.

   .. grid-item-card:: Computation Models
      :link: models/index
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Understand the CLF and CBC synthesis methods for
      each of the four supported system classes.

   .. grid-item-card:: Benchmark Cases
      :link: cases/index
      :link-type: doc
      :class-card: sd-border-0 sd-shadow-sm

      Explore the 17 built-in benchmark case studies with
      pre-configured trajectory data.

----

Supported System Classes
========================

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Continuous-Time Linear (ct-LS)
      :class-card: sd-border-0 sd-shadow-sm

      Linear systems :math:`\dot{x} = Ax + Bu` with unknown :math:`A, B`.
      SDP-based CLF/CBC synthesis via PICOS.

   .. grid-item-card:: Continuous-Time Nonlinear Polynomial (ct-NPS)
      :class-card: sd-border-0 sd-shadow-sm

      Polynomial systems :math:`\dot{x} = f(x) + g(x)u` with unknown dynamics.
      SOS-based synthesis using monomial lifting.

   .. grid-item-card:: Discrete-Time Linear (dt-LS)
      :class-card: sd-border-0 sd-shadow-sm

      Linear systems :math:`x_{k+1} = Ax_k + Bu_k` with unknown :math:`A, B`.
      Schur complement LMI formulation.

   .. grid-item-card:: Discrete-Time Nonlinear Polynomial (dt-NPS)
      :class-card: sd-border-0 sd-shadow-sm

      Polynomial systems :math:`x_{k+1} = f(x_k) + g(x_k)u_k`.
      Requires user-specified monomials and :math:`\Theta(x)` matrix.

----

Related Paper
=============

This work was accepted to the Hybrid Systems Computation and Control (HSCC)
Conference 2025.

.. code-block:: bibtex

   @inproceedings{TRUST2025,
     title={TRUST: StabiliTy and Safety ContRoller Synthesis for Unknown
            Dynamical Models Using a Single Trajectory},
     author={Gardner, Jamie and Wooding, Ben and Nejati, Amy and Lavaei, Abolfazl},
     booktitle={Proceedings of the 28th ACM International Conference on
                Hybrid Systems: Computation and Control},
     pages={1--16},
     year={2025}
   }

- `ACM Digital Library <https://dl.acm.org/doi/10.1145/3716863.3718036>`_
- `arXiv preprint <https://arxiv.org/abs/2503.08081>`_


.. toctree::
   :hidden:
   :caption: Getting Started

   installation
   usage

.. toctree::
   :hidden:
   :caption: Computation Models

   models/index

.. toctree::
   :hidden:
   :caption: Reference

   cases/index
