"""
Module for rate limiting utilities.

This module demonstrates how to create an in-memory rate limiter using the
InMemoryRateLimiter from langchain_core. It limits requests to one per second,
with a check interval of 0.1 seconds and a maximum bucket size of 2.
"""

from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(requests_per_second=1, check_every_n_seconds=0.1, max_bucket_size=2)  # pylint: disable=line-too-long  # "black" is reformatting these lines
