# Profiling

xDEM has a built-in profiling tool, that can be used to provide more insight on the memory and time use of a function if needed.
```{warning}
The profiling functionalities rely on psutil and plotly, which can be installed manually or using xDEM development dependencies.
```

With the profiling activated with the graphs output, two kinds of .HTML graphs will be created by default :
* an icicle graph `time_graph.html`, showing the time spent in each step of the entire process
* a graph `memory_[function].html` for each decorated functions used, showing the memory consumption of xDEM at regular intervals during the execution

## Configuration and parameters

xDEM's profiling configuration works just like a pipeline step. It is executed only if at least one of this two parameters is set to True :

| Name              | Description                       | Type | Default value | Required |
|-------------------|-----------------------------------| ------- | ------- | ------- |
| **save_graphs**   | Save the default graphs generated | bool | False | No |
| **save_raw_data** | Save the raw data on calls as a .pickle file | bool | False | No |

Example of initialization:

```{code-cell} ipython3
from xdem.profiler import Profiler

Profiler.enable(save_graphs=True, save_raw_data=True)
```

After this, if `save_graphs` or `save_raw_data` are True, every profiled function will be studied by the profiler.

## Saved profiling data

When *save_raw_data* is enabled, xDEM saves the profiling information as a .pickle file containing a {class}`~pandas.DataFrame` with the following structure:

| Name              | Description                                                                                                                                       |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| **level**         | Depth of the function call in the profiling stack                                                                                                 |
| **uuid_function** | Unique universal identifier (UUID) of the function call                                                                                           |
| **name**         | Understandable name given to the function call                                                                                                    |
| **uuid_parent**   | UUID of the "parent" call (call that was running when this call was made)                                                                         |
| **time**          | Time (in seconds) it took to execute the function                                                                                                 |
| **call_time**     | Timestamp (in seconds) at which the call was made                                                                                                 |
| **memory**        | Either None or a list of (timestamp memory) tuples representing memory consumption (in megabytes) at each timestamp during the function execution |


## The profiled functions

Currently, some processes are already profiled by xDEM with a memory consumption report each 0.05 seconds.
- all the terrain attributes computation {class}`xdem.DEM` attributes computations
- all the co-registration processing through the {class}`xdem.Coreg.fit_and_apply` function

### Modifying the profiled functions

To profile other functions and add them to the summary graphs and data, simply add the *@profile* decorator before them, providing a descriptive name.

If you also want to track memory usage over time for a specific function call, set `memprof=True` in the decorator.
If the function is too fast (or slow) for the default memory sampling interval, you can modify it with *interval* (in seconds).

```{code-cell} ipython3
from xdem.profiler import Profiler

@profile("my profiled function", memprof=True, interval=0.5)  # type: ignore
def my_function():
    ...
```

| Name         | Description                                      | Type  | Default value | Required |
|--------------|--------------------------------------------------|-------|---------------|----------|
| **name**     | Name of the function in the report               | str   |               | Yes      |
| **interval** | Memory sampling interval (seconds)               | float | 0.05          | No       |
| **memprof**  | Whether to profile the memory consumption or not | bool  | False         | No       |


### Output graphs example

Here are two examples of graphs with a personal profiled function `my_program`, containing the computation of a few attributes and a co-registration:

![time_graph.html](imgs/profiling_time_graph.html.png)
![memory_my_program.html](imgs/profiling_memory_my_program.html.png)
