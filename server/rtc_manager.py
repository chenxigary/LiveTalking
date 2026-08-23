###############################################################################
#  WebRTC 连接管理 + RTC 音频/视频接收
###############################################################################

import json
import asyncio
import random
import copy
from typing import Dict, Optional
import queue

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender

from utils.logger import logger


# def _rand_session_id(n: int = 6) -> int:
#     """生成 N 位随机 session ID"""
#     return random.randint(10 ** (n - 1), 10 ** n - 1)


from server.session_manager import session_manager
from server.session_manager import MaxSessionError

class RTCManager:
    """
    WebRTC 连接管理器。
    
    管理 PeerConnection 生命周期、音视频轨道收发、DataChannel。
    """

    def __init__(self, opt):
        """
        Args:
            opt: 全局配置
        """
        self.opt = opt
        self.pcs: set = set()
        self._pcs_by_session: Dict[str, RTCPeerConnection] = {}
        self._session_by_pc: Dict[RTCPeerConnection, str] = {}
        self._closing_pcs: set = set()
        self._disconnect_tasks: Dict[RTCPeerConnection, asyncio.Task] = {}
        self._disconnect_grace_sec = max(
            0.0, float(getattr(opt, "webrtc_disconnect_grace_sec", 10.0))
        )

    def _cancel_disconnect_cleanup(self, pc):
        task = self._disconnect_tasks.pop(pc, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()

    async def _cleanup_pc(self, pc, sessionid: str, reason: str) -> bool:
        """Idempotently close one peer and release its render session."""

        if pc in self._closing_pcs:
            return False
        self._closing_pcs.add(pc)
        self._cancel_disconnect_cleanup(pc)
        logger.info("Closing WebRTC session %s (%s)", sessionid, reason)
        try:
            if getattr(pc, "connectionState", None) != "closed":
                await pc.close()
        finally:
            self.pcs.discard(pc)
            if self._pcs_by_session.get(sessionid) is pc:
                self._pcs_by_session.pop(sessionid, None)
            self._session_by_pc.pop(pc, None)
            session_manager.remove_session(sessionid)
            self._closing_pcs.discard(pc)
        return True

    def _schedule_disconnect_cleanup(self, pc, sessionid: str, source: str):
        """Give transient ICE disconnects a short recovery window."""

        self._cancel_disconnect_cleanup(pc)

        async def expire_disconnected():
            try:
                await asyncio.sleep(self._disconnect_grace_sec)
                connection_state = getattr(pc, "connectionState", None)
                ice_state = getattr(pc, "iceConnectionState", None)
                if connection_state == "disconnected" or ice_state == "disconnected":
                    await self._cleanup_pc(
                        pc,
                        sessionid,
                        f"{source} disconnected for {self._disconnect_grace_sec:g}s",
                    )
            except asyncio.CancelledError:
                return
            finally:
                if self._disconnect_tasks.get(pc) is asyncio.current_task():
                    self._disconnect_tasks.pop(pc, None)

        self._disconnect_tasks[pc] = asyncio.create_task(expire_disconnected())

    def _register_pc(self, pc, sessionid: str):
        self.pcs.add(pc)
        self._pcs_by_session[sessionid] = pc
        self._session_by_pc[pc] = sessionid

        async def observe_state(state: str, source: str):
            logger.info("%s state is %s for session %s", source, state, sessionid)
            if state in ("connected", "completed"):
                self._cancel_disconnect_cleanup(pc)
            elif state == "disconnected":
                self._schedule_disconnect_cleanup(pc, sessionid, source)
            elif state in ("failed", "closed"):
                await self._cleanup_pc(pc, sessionid, f"{source} {state}")

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            await observe_state(pc.connectionState, "connection")

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            await observe_state(pc.iceConnectionState, "ICE connection")

    async def _create_pc_and_answer(self, avatar_session, sessionid, offer):
        """创建 PeerConnection、添加轨道、SDP 交换，返回已完成 answer 的 pc"""
        ice_server = RTCIceServer(urls=self.opt.stun)
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=[ice_server])
        )
        self._register_pc(pc, sessionid)

        try:
            # 添加发送轨道
            from server.webrtc import HumanPlayer
            player = HumanPlayer(avatar_session)
            pc.addTrack(player.audio)
            pc.addTrack(player.video)

            # 设置编解码器偏好
            capabilities = RTCRtpSender.getCapabilities("video")
            if capabilities and hasattr(capabilities, "codecs"):
                get_name = lambda c: getattr(c, "name", "") or getattr(c, "mimeType", "")
                preferences = [c for c in capabilities.codecs if "H264" in get_name(c)]
                preferences += [c for c in capabilities.codecs if "VP8" in get_name(c)]
                preferences += [c for c in capabilities.codecs if "rtx" in get_name(c)]
                transceivers = pc.getTransceivers()
                video_transceiver = next((t for t in transceivers if t.kind == "video"), None)
                if video_transceiver:
                    video_transceiver.setCodecPreferences(preferences)

            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
        except Exception:
            await self._cleanup_pc(pc, sessionid, "offer setup failed")
            raise

        return pc

    async def close_session(self, sessionid: str, reason: str = "client request") -> bool:
        """Close a specific WebRTC peer; safe to call more than once."""

        pc = self._pcs_by_session.get(sessionid)
        if pc is not None:
            return await self._cleanup_pc(pc, sessionid, reason)
        if session_manager.has_session(sessionid):
            session_manager.remove_session(sessionid)
            return True
        return False

    async def handle_offer(self, request):
        """处理 WebRTC offer 信令"""
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        try:
            sessionid = await session_manager.create_session(params)
        except MaxSessionError as e:
            logger.warning("Rejecting offer: %s", e)
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": str(e)}),
            )
        logger.info('offer sessionid=%s', sessionid)

        pc = await self._create_pc_and_answer(
            session_manager.get_session(sessionid), sessionid, offer
        )

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
                "sessionid": sessionid,
            }),
        )

    async def handle_whep(self, request):
        """
        处理 WHEP 信令（WebRTC HTTP Egress Protocol）

        - 请求 body 为裸 SDP offer（Content-Type: application/sdp）
        - 扩展参数通过 query string 传入（avatar, tts, tts_server 等）
        - 返回 SDP answer（Content-Type: application/sdp）
        - sessionid 通过 X-Session-ID 响应头返回
        """
        params = dict(request.query)
        # 客户端可通过 query param 自定义 sessionid，不传则自动生成
        client_sid = params.pop("sessionid", None)

        offer_sdp = await request.text()
        offer = RTCSessionDescription(sdp=offer_sdp, type="offer")

        try:
            sessionid = await session_manager.create_session(params, sessionid=client_sid)
        except MaxSessionError as e:
            logger.warning("Rejecting whep: %s", e)
            return web.Response(
                status=503,
                content_type="text/plain",
                text=str(e),
            )
        logger.info("whep sessionid=%s", sessionid)

        pc = await self._create_pc_and_answer(
            session_manager.get_session(sessionid), sessionid, offer
        )

        return web.Response(
            status=201,
            content_type="application/sdp",
            text=pc.localDescription.sdp,
            headers={"X-Session-ID": sessionid},
        )

    async def handle_rtcpush(self, push_url, sessionid: str):
        """RTCPush 模式：主动推流"""
        import aiohttp
        await session_manager.create_session({}, sessionid)
        avatar_session = session_manager.get_session(sessionid)

        pc = RTCPeerConnection()
        self._register_pc(pc, sessionid)

        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        await pc.setLocalDescription(await pc.createOffer())

        async with aiohttp.ClientSession() as session:
            async with session.post(push_url, data=pc.localDescription.sdp) as response:
                answer_sdp = await response.text()

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer_sdp, type='answer')
        )

    async def shutdown(self):
        """关闭所有 PeerConnection"""
        peers = list(self.pcs)
        await asyncio.gather(*(
            self._cleanup_pc(
                pc,
                self._session_by_pc.get(pc, ""),
                "server shutdown",
            )
            for pc in peers
        ))
        for task in list(self._disconnect_tasks.values()):
            task.cancel()
        self._disconnect_tasks.clear()
