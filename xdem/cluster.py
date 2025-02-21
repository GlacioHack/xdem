# Copyright (c) 2024 xDEM developers
#
# This file is part of the xDEM project:
# https://github.com/glaciohack/xdem
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module defines the cluster configurations."""

import multiprocessing
from multiprocessing.pool import Pool
from typing import Any, Callable, Dict, List, Optional


class ClusterGenerator:
    def __new__(cls, name: str, nb_workers: int = 2) -> "AbstractCluster":  # type: ignore
        if name == "basic":
            cluster: AbstractCluster = BasicCluster()
        else:
            cluster = MpCluster(conf={"nb_workers": nb_workers})
        return cluster


class AbstractCluster:
    def __init__(self) -> None:
        self.pool: Optional[Pool] = None

    def __enter__(self) -> "AbstractCluster":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        pass

    def launch_task(
        self, fun: Callable[..., Any], args: Optional[List[Any]] = None, kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        pass

    def get_res(self, future: Any) -> Any:
        return future

    def return_wrapper(self) -> None:
        pass

    def tile_retriever(self, res: Any) -> None:
        pass


class BasicCluster(AbstractCluster):
    def launch_task(
        self, fun: Callable[..., Any], args: Optional[List[Any]] = None, kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        return fun(*args, **kwargs)


class MpCluster(AbstractCluster):
    def __init__(self, conf: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        nb_workers = 1
        if conf is not None:
            nb_workers = conf.get("nb_workers", 1)
        ctx_in_main = multiprocessing.get_context("forkserver")
        self.pool = ctx_in_main.Pool(processes=nb_workers, maxtasksperchild=10)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()

    def launch_task(
        self, fun: Callable[..., Any], args: Optional[List[Any]] = None, kwargs: Optional[Dict[str, Any]] = None
    ) -> Any:
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        if self.pool is not None:
            return self.pool.apply_async(fun, args=args, kwds=kwargs)

    def get_res(self, future: Any) -> Any:
        return future.get(timeout=5000)
