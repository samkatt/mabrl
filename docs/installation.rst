.. highlight:: shell

============
Installation
============

.. Stable release
.. --------------

.. To install multi-agent Bayesian reinforcement learning, run this command in your terminal:

.. .. code-block:: console

..     $ pip install mabrl

.. This is the preferred method to install multi-agent Bayesian reinforcement learning, as it will always install the most recent stable release.

.. If you don't have `pip`_ installed, this `Python installation guide`_ can guide
.. you through the process.

.. .. _pip: https://pip.pypa.io
.. .. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for multi-agent Bayesian reinforcement learning can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/samkatt/mabrl

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/samkatt/mabrl/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install

Currently the only dependency that is not automatically installed is
`general_bayes_adaptive_pomdps`, which must be in python's path for this
package to work.

.. _Github repo: https://github.com/samkatt/mabrl
.. _tarball: https://github.com/samkatt/mabrl/tarball/master
