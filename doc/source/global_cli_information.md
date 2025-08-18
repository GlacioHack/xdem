(global_cli_information)=

# Command line interface (CLI)

To simplify the use of xDEM and provide a universal tool to assess one or multiple DEMs,
we have decided to implement a Command Line Interface (CLI).
To support this, we offer a set of workflows that can be easily run using a configuration file.
Users can also create their own workflows and submit them for inclusion.

```{note}
**workflow definition** : combinations of various xDEM and GeoUtils features tailored to specific applications
```

All workflows follow the same command structure:
```{code}
xdem workflow_name --config config_file.yaml
```

The YAML configuration file contains at least the required input parameters for the workflow.
xDEM then automatically fills in the rest with suggested settings. Users are free to edit the
configuration file to run only the parts they need.


To display a template of all available configuration options for the YAML file, use the following command:

```{code}
xdem workflow_name --display-template-config
```

```{note}
At the end of the execution, several output files are saved to disk, including an HTML report
and its corresponding PDF version.
```

```{toctree}
:caption: Available workflows
:maxdepth: 2
cli_topo
cli_accuracy
```
