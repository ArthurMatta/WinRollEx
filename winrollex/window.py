from datetime import timedelta
from math import floor
from typing import Iterable

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

TimeDelta = relativedelta | pd.Timedelta | timedelta


class Window:
    def __init__(self, data: pd.DataFrame | np.ndarray) -> None:
        """
        A class to generate training and test sets based on a rolling (a.k.a moving) or
        expanding (a.k.a. incremental) window.

        Args:
            data: The input data.
        """
        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self._size = data.shape[0]

    def _window_split(
        self,
        training_window_size: int | float,
        test_window_size: int | float | None = None,
        window_increment: int | float | None = None,
        iterations: int | None = None,
    ) -> Iterable[tuple[int, int, int]]:
        if not isinstance(training_window_size, (int, float)):
            raise ValueError(
                "Training window size must be an integer or a float.",
            )
        if not isinstance(test_window_size, (int, float, type(None))):
            raise ValueError(
                "Test window size must be an integer, a float, or None.",
            )
        if not isinstance(window_increment, (int, float, type(None))):
            raise ValueError(
                "Window increment must be an integer, a float, or None.",
            )
        if not isinstance(iterations, (int, type(None))):
            raise ValueError("Iterations must be an integer or None.")

        trw = (
            training_window_size
            if isinstance(training_window_size, int)
            else floor(self._size * training_window_size)
        )

        if trw <= 0 or trw > self._size:
            raise ValueError(
                "Training window size must be greater than 0 and represent a"
                "value (concrete or percentage) less than or equal to the"
                "size of the data.",
            )

        if iterations is not None:
            if iterations < 1:
                raise ValueError(
                    "Iterations must be greater than or equal to 1.",
                )
            leftover = self._size - trw
            tsw = leftover // iterations
            wi = tsw
        elif test_window_size is not None:
            tsw = (
                test_window_size
                if isinstance(test_window_size, int)
                else floor(self._size * test_window_size)
            )
            if tsw <= 0 or tsw > self._size:
                raise ValueError(
                    "Test window size must be greater than 0 and represent a"
                    "value (concrete or percentage) less than or equal to the"
                    "size of the data.",
                )

            wi = (
                (
                    window_increment
                    if isinstance(window_increment, int)
                    else floor(self._size * window_increment)
                )
                if window_increment is not None
                else tsw
            )

            iterations = 1 + (self._size - (trw + tsw)) // wi
        else:
            raise ValueError(
                "Either the number of iterations or the test window"
                "must be provided.",
            )

        if trw + tsw > self._size:
            raise ValueError(
                "The sum of the training and test windows must be less "
                "than or equal to the size of the data.",
            )

        for it in range(iterations):
            lb = wi * it
            mb = trw + lb
            ub = tsw + mb
            yield lb, mb, ub

    def rolling(
        self,
        training_window_size: int | float,
        test_window_size: int | float | None = None,
        window_increment: int | float | None = None,
        iterations: int | None = None,
    ) -> Iterable[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        This function creates a generator that yields training and test sets
        based on a rolling window.

        Args:
            training_window_size (int | float): The size of the training
                window. Can be either an integer representing the number of
                rows or a float representing a percentage of the total
                number of rows.
            test_window_size (int | float | None): The size of the test window.
                Can be either an integer representing the number of rows or a
                float representing a percentage of the total number of rows.
                Either `test_window_size` or `iterations` must be provided.
            window_increment (int | float | None): The increment of the window.
                Can be either an integer representing the number of rows or a
                float representing a percentage of the total number of rows.
                If None, the window increment will be the same as the test
                window size.
            iterations (int | None): The number of iterations. Either
                `iterations` or `test_window_size` must be provided. If not
                None, `test_window_size` and `window_increment` will be
                ignored.

        Yields:
            tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the training
                and test windows.

        """
        for lb, mb, ub in self._window_split(
            training_window_size,
            test_window_size,
            window_increment,
            iterations,
        ):
            tr = self.data.iloc[lb:mb]
            ts = self.data.iloc[mb:ub]
            yield tr, ts

    def expanding(
        self,
        training_window_size: int | float,
        test_window_size: int | float | None = None,
        window_increment: int | float | None = None,
        iterations: int | None = None,
    ) -> Iterable[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        This function creates a generator that yields training and test sets
        based on a expanding window.

        Args:
            training_window_size (int | float): The size of the training
                window. Can be either an integer representing the number of
                rows or a float representing a percentage of the total
                number of rows.
            test_window_size (int | float | None): The size of the test window.
                Can be either an integer representing the number of rows or a
                float representing a percentage of the total number of rows.
                Either `test_window_size` or `iterations` must be provided.
            window_increment (int | float | None): The increment of the window.
                Can be either an integer representing the number of rows or a
                float representing a percentage of the total number of rows.
                If None, the window increment will be the same as the test
                window size.
            iterations (int | None): The number of iterations. The actual
                number of iterations will be one more than this to include
                the first training window. Either `iterations` or
                `test_window_size` must be provided. If not None,
                `test_window_size` will be ignored.

        Yields:
            tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the training
                and test windows.
        """
        for _, mb, ub in self._window_split(
            training_window_size,
            test_window_size,
            window_increment,
            iterations,
        ):
            tr = self.data.iloc[:mb]
            ts = self.data.iloc[mb:ub]
            yield tr, ts

    def _temporal_window_split(
        self,
        training_window_size: TimeDelta | float,
        test_window_size: TimeDelta | float | None = None,
        timedelta: TimeDelta | None = None,
        iterations: int | None = None,
    ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
        if not isinstance(training_window_size, (TimeDelta, float)):
            raise ValueError(
                "Training window size must be a TimeDelta or float.",
            )
        if not isinstance(
            test_window_size,
            (int, float, TimeDelta, type(None)),
        ):
            raise ValueError(
                "Test window size must be a TimeDelta, a float or None.",
            )
        if not isinstance(timedelta, (TimeDelta, type(None))):
            raise ValueError(
                "Window increment must be a TimeDelta or None.",
            )
        if not isinstance(iterations, (int, type(None))):
            raise ValueError("Iterations must be an integer or None.")

        start_date = pd.to_datetime(self.data.index.min())
        end_date = pd.to_datetime(self.data.index.max())
        date_diff = end_date - start_date

        trw = (
            training_window_size
            if isinstance(training_window_size, TimeDelta)
            else training_window_size * date_diff
        )

        if iterations is not None:
            leftover = date_diff - trw
            tsw = leftover // iterations
            wi = tsw
        elif test_window_size is not None:
            tsw = (
                test_window_size
                if isinstance(test_window_size, TimeDelta)
                else test_window_size * date_diff
            )

            wi = timedelta if timedelta is not None else tsw

            iterations = 1 + (date_diff - (trw + tsw)) // wi
        else:
            raise ValueError(
                "Either the number of iterations or the test window"
                "must be provided.",
            )

        for it in range(iterations):
            lb = start_date + it * wi
            mb = lb + trw
            ub = mb + tsw
            yield lb, mb, ub

    def temporal_expanding(
        self,
        training_window_size: TimeDelta | float,
        test_window_size: TimeDelta | float | None = None,
        timedelta: TimeDelta | None = None,
        iterations: int | None = None,
    ) -> Iterable[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        This function creates a generator that yields training and test sets
        based on a temporal expanding window. It assumes the index of the data
        is a datetime index.

        Args:
            training_window_size (TimeDelta | float): The size of the training
                window. Can be either a pandas.TimeDelta, a datetime.deltatime,
                a dateutil.relativedelta, or a float representing a percentage 
                of the total number of rows.
            test_window_size (TimeDelta | float | None): The size of the test window.
                Can be either a pandas.TimeDelta, a datetime.deltatime,
                a dateutil.relativedelta, or a float representing a percentage 
                of the total number of rows. Either `test_window_size` or 
                `timedelta` must be provided.
            timedelta (TimeDelta | None): The increment of the window.
                Can be either a pandas.TimeDelta, a datetime.deltatime or
                a dateutil.relativedelta. If None, the timedelta will be the 
                same as the `test_window_size`.
            iterations (int | None): The number of iterations. Either
                `iterations` or `test_window_size` must be provided. If not
                None, `test_window_size` and `timedelta` will be ignored.

        Yields:
            tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the training
                and test windows.

        """
        for _, mb, ub in self._temporal_window_split(
            training_window_size,
            test_window_size,
            timedelta,
            iterations,
        ):
            tr = self.data.loc[:mb]
            ts = self.data.loc[mb:ub]
            yield tr, ts

    def temporal_rolling(
        self,
        training_window_size: TimeDelta | float,
        test_window_size: TimeDelta | float | None = None,
        timedelta: TimeDelta | None = None,
        iterations: int | None = None,
    ) -> Iterable[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        This function creates a generator that yields training and test sets
        based on a temporal rolling window. It assumes the index of the data
        is a datetime index.

        Args:
            training_window_size (TimeDelta | float): The size of the training
                window. Can be either a pandas.TimeDelta, a datetime.deltatime,
                a dateutil.relativedelta, or a float representing a percentage 
                of the total number of rows.
            test_window_size (TimeDelta | float | None): The size of the test window.
                Can be either a pandas.TimeDelta, a datetime.deltatime,
                a dateutil.relativedelta, or a float representing a percentage 
                of the total number of rows. Either `test_window_size` or 
                `timedelta` must be provided.
            timedelta (TimeDelta | None): The increment of the window.
                Can be either a pandas.TimeDelta, a datetime.deltatime or
                a dateutil.relativedelta. If None, the timedelta will be the 
                same as the `test_window_size`.
            iterations (int | None): The number of iterations. Either
                `iterations` or `test_window_size` must be provided. If not
                None, `test_window_size` and `timedelta` will be ignored.

        Yields:
            tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the training
                and test windows.

        """
        for lb, mb, ub in self._temporal_window_split(
            training_window_size,
            test_window_size,
            timedelta,
            iterations,
        ):
            tr = self.data.loc[lb:mb]
            ts = self.data.loc[mb:ub]
            yield tr, ts
