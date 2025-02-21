""" Functions to test the clusters."""

import time

import pytest

from xdem.cluster import BasicCluster, ClusterGenerator, MpCluster


# Sample function for testing
def sample_function(x: float, y: float) -> float:
    return x + y


# Function to simulate a long-running task
def long_running_task(x: float) -> float:
    time.sleep(1)
    return x * 2


class TestClusterGenerator:
    def test_basic_cluster(self) -> None:
        # Test that tasks are run synchronously in BasicCluster
        cluster = ClusterGenerator(name="basic")
        assert isinstance(cluster, BasicCluster)

        result = cluster.launch_task(sample_function, args=[2, 3])
        assert result == 5

    def test_mp_cluster_task(self) -> None:
        # Test that tasks are launched asynchronously in MpCluster
        cluster = ClusterGenerator("multiprocessing", nb_workers=2)
        assert isinstance(cluster, MpCluster)

        future = cluster.launch_task(sample_function, args=[2, 3])
        result = cluster.get_res(future)
        assert result == 5

    def test_mp_cluster_parallelism(self) -> None:
        # Test that multiple tasks are run in parallel
        cluster = ClusterGenerator("multiprocessing", nb_workers=2)
        assert isinstance(cluster, MpCluster)

        futures = [cluster.launch_task(long_running_task, args=[i]) for i in range(4)]
        results = [cluster.get_res(f) for f in futures]
        assert results == [0, 2, 4, 6]

    def test_mp_cluster_termination(self) -> None:
        # Test that the pool terminates correctly after closing
        cluster = ClusterGenerator("multiprocessing", nb_workers=2)
        assert isinstance(cluster, MpCluster)

        # Close the cluster
        cluster.close()

        # Expect an error when trying to launch a task after closing
        with pytest.raises(ValueError):
            cluster.launch_task(sample_function, args=[2, 3])
