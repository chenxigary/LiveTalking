###############################################################################
#  全局会话管理器 (Session Manager)
###############################################################################

import asyncio
import uuid
from typing import Dict, Optional
from utils.logger import logger
from avatars.base_avatar import BaseAvatar


class MaxSessionError(Exception):
    """会话数达到上限时抛出"""
    pass

def _rand_session_id() -> str:
    """生成 UUID session ID"""
    return str(uuid.uuid4())

class SessionManager:
    """
    全局数字人会话管理器。
    
    统一管理 avatar_sessions 生命周期，并在脱离 WebRTC 时依然保持服务可用。
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.sessions: Dict[str, BaseAvatar] = {}
            self.build_session_fn = None
            self.max_session = 1   # default, override via set_max_session()
            self.initialized = True
        if not hasattr(self, "_pending_sessions"):
            self._pending_sessions = set()

    def set_max_session(self, n: int):
        """设置最大并发会话数"""
        self.max_session = max(1, n)

    def init_builder(self, build_session_fn):
        """配置用于构建 avatar_session 的工厂函数"""
        self.build_session_fn = build_session_fn
        
    def get_session(self, sessionid: str) -> Optional[BaseAvatar]:
        """获取已存活的会话"""
        return self.sessions.get(sessionid)

    def has_session(self, sessionid: str) -> bool:
        """检查会话是否存在"""
        return sessionid in self.sessions and self.sessions[sessionid] is not None
        
    async def create_session(self, params: dict, sessionid: str = None) -> str:
        """
        在异步环境中创建一个新会话
        如果 sessionid 为 None，则自动生成。
        """
        if self.build_session_fn is None:
            raise Exception("SessionManager builder not initialized")
            
        if sessionid is None:
            sessionid = _rand_session_id()
            
        # Count in-flight reservations as well as fully-built sessions.  The
        # builder runs in an executor, so concurrent /offer requests can all
        # reach this point while earlier builds are still running.  Ignoring
        # those reservations allowed max_session=5 to admit six or more
        # simultaneous Wav2Lip sessions during a browser reconnect burst.
        active_count = len(self.sessions) + len(self._pending_sessions)
        if active_count >= self.max_session:
            raise MaxSessionError(
                f"Maximum session limit reached ({active_count}/{self.max_session})"
            )

        if sessionid in self.sessions or sessionid in self._pending_sessions:
            raise MaxSessionError(f"Session already exists: {sessionid}")

        logger.info('Creating sessionid=%s, current session num=%d', sessionid, active_count)
        self._pending_sessions.add(sessionid)

        # 在线程池中构建 session（加载模型非常耗时）
        build_future = asyncio.get_running_loop().run_in_executor(
            None, self.build_session_fn, sessionid, params
        )
        try:
            # Shield the executor future so an abandoned HTTP request cannot
            # cancel the bookkeeping future while its worker keeps consuming
            # resources.  The reservation is released only when that worker
            # has actually stopped.
            avatar_session = await asyncio.shield(build_future)
        except asyncio.CancelledError:
            def release_abandoned_build(future):
                self._pending_sessions.discard(sessionid)
                try:
                    future.result()
                except BaseException:
                    pass

            build_future.add_done_callback(release_abandoned_build)
            raise
        except BaseException:
            # A failed build must release its reservation or all later offers
            # can be rejected indefinitely even though no usable session was
            # created.
            self._pending_sessions.discard(sessionid)
            raise
        self._pending_sessions.discard(sessionid)
        self.sessions[sessionid] = avatar_session
        return sessionid
        
    def add_session(self, sessionid: str, avatar_session: BaseAvatar):
        """同步添加静态或外部管理的会话（供非服务端入口调用）"""
        self._pending_sessions.discard(sessionid)
        self.sessions[sessionid] = avatar_session
        
    def remove_session(self, sessionid: str):
        """销毁会话资源"""
        self._pending_sessions.discard(sessionid)
        if sessionid in self.sessions:
            logger.info(f"Removing session {sessionid}")
            # todo: 还可以主动调 avatar_session 释放
            self.sessions.pop(sessionid, None)

# 单例抛出
session_manager = SessionManager()
