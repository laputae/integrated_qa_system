import time
from typing import Optional

from mysql_qa import RedisClient


class RateLimiter:
    def __init__(self, redis_client: Optional[RedisClient] = None):
        self._redis = redis_client

    @property
    def redis(self):
        if self._redis is None:
            self._redis = RedisClient()
        return self._redis

    def _rate_limit_key(self, prefix: str, identifier: str) -> str:
        window = int(time.time()) // 60
        return f"rate_limit:{prefix}:{identifier}:{window}"

    def check(self, key_prefix: str, identifier: str,
              limit: int, window_seconds: int = 60) -> bool:
        key = self._rate_limit_key(key_prefix, identifier)
        try:
            count = self.redis.client.incr(key)
            if count == 1:
                self.redis.client.expire(key, window_seconds)
            return count <= limit
        except Exception:
            return True

    def check_login_limit(self, ip_address: str) -> bool:
        return self.check("login", ip_address, limit=5, window_seconds=60)

    def check_register_limit(self, ip_address: str) -> bool:
        return self.check("register", ip_address, limit=3, window_seconds=3600)

    def check_query_limit(self, user_id: int) -> bool:
        return self.check("query", str(user_id), limit=30, window_seconds=60)

    def check_stream_limit(self, user_id: int) -> bool:
        return self.check("stream", str(user_id), limit=10, window_seconds=60)
