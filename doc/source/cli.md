(cli)=

# Command line interface

xDEM provides a **command line interface (CLI)** that allows to run **workflows** for various analyses relevant to elevation data.

**Workflows** consist of a series of analysis steps often performed together, that typically fit into three broad categories:
- File opening and pre-preprocessing steps (e.g. reprojection to same coordinate system, interpolation to the same points), with option to perform those **out-of-memory** on large datasets,
- Main analysis using one or several of xDEM features (e.g., co-registration and uncertainty quantification to perform an accuracy assessment), using **best-performing methods** by default,
- Generation and writing of outputs needed to interpret the analysis (e.g., files, tables, figures), including **a general report** provided as an HTML output.

```{caution}
The CLI and workflows are currently experimental! We are adapting their early interface with feedback and improving performance for future releases.
```

## Basic usage

The CLI relies on a **configuration file**, easily editable, and which comes pre-set with defaults that we recommend for general cases.

All workflows are run using the following command structure, replacing the argument `workflow_name` by that of an existing workflow, such as `accuracy` or `topo`:

```{code}
xdem workflow_name --config config_file.yaml
```

Optionally, the output folder can be set directly in the command-line using `--output` (overrides YAML file).

The configuration file is written as a YAML. To display a template of all available configuration options for the YAML file, use the following command:

```{code}
xdem workflow_name --template-config
```

Optionally, a output path to save the template can be set directly in the command-line using `--output`.

When edited by a user, a configuration file **must contain at minima the input parameters listed as "required"** on the documentation page of the given workflow.
xDEM then automatically fills in the rest with default settings. Users are free to edit the configuration file to run only the parts they need.

```{note}
:class: tip
:class: margin

We welcome users to create their own workflows and submit them for inclusion in xDEM!
```

```{toctree}
:caption: Workflows
:maxdepth: 2
cli_accuracy
cli_topo
```
