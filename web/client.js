var pc = null;
var sessionHeartbeatTimer = null;
var sessionHeartbeatFailures = 0;
var sessionHeartbeatSerial = 0;

function stopSessionHeartbeat() {
    sessionHeartbeatSerial += 1;
    if (sessionHeartbeatTimer !== null) {
        clearInterval(sessionHeartbeatTimer);
        sessionHeartbeatTimer = null;
    }
    sessionHeartbeatFailures = 0;
}

function startSessionHeartbeat(sessionId) {
    stopSessionHeartbeat();
    var heartbeatSerial = sessionHeartbeatSerial;
    var endpoint = ((window.location.origin && window.location.origin !== 'null')
        ? window.location.origin : 'http://localhost:8010') + '/api/session/heartbeat';
    var publishHeartbeat = (ok, error) => window.dispatchEvent(new CustomEvent(
        'xiaoman-avatar-heartbeat',
        {detail: {
            ok: Boolean(ok),
            session_id: String(sessionId),
            failures: sessionHeartbeatFailures,
            error: error || null,
        }}
    ));
    var renew = () => fetch(endpoint, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({sessionid: sessionId}),
    }).then(async (response) => {
        if (heartbeatSerial !== sessionHeartbeatSerial) return;
        if (!response.ok) throw new Error('heartbeat HTTP ' + response.status);
        var payload = await response.json();
        if (heartbeatSerial !== sessionHeartbeatSerial) return;
        if (Number(payload.code) !== 0) {
            throw new Error(payload.msg || 'heartbeat session rejected');
        }
        sessionHeartbeatFailures = 0;
        publishHeartbeat(true, null);
    }).catch((error) => {
        if (heartbeatSerial !== sessionHeartbeatSerial) return;
        sessionHeartbeatFailures += 1;
        publishHeartbeat(false, error && error.message ? error.message : String(error));
    });
    renew();
    sessionHeartbeatTimer = setInterval(renew, 3000);
}

function negotiate() {
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });
    return pc.createOffer().then((offer) => {
        return pc.setLocalDescription(offer);
    }).then(() => {
        // wait for ICE gathering to complete
        return new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                const checkState = () => {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(() => {
        var offer = pc.localDescription;
        var endpoint = (window.location.origin && window.location.origin !== 'null')
            ? window.location.origin + '/offer'
            : 'http://localhost:8010/offer';
        return fetch(endpoint, {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
                session_ttl_sec: 30,
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then((response) => {
        if (!response.ok) {
            throw new Error('Server returned ' + response.status + ' ' + response.statusText);
        }
        return response.json();
    }).then((answer) => {
        var sidEl = document.getElementById('sessionid');
        if (sidEl) sidEl.value = answer.sessionid;
        startSessionHeartbeat(String(answer.sessionid));
        return pc.setRemoteDescription(answer);
    });
}

function start() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    var useStunEl = document.getElementById('use-stun');
    if (useStunEl && useStunEl.checked) {
        config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
    }

    pc = new RTCPeerConnection(config);

    // connect audio / video
    pc.addEventListener('track', (evt) => {
        if (evt.track.kind == 'video') {
            var vEl = document.getElementById('video');
            if (vEl) {
                // Keep audio and video on separate elements.  aiortc may put
                // both tracks in evt.streams[0], which can make an unmuted
                // video fail the browser's autoplay policy and render black.
                vEl.srcObject = new MediaStream([evt.track]);
                vEl.muted = true;
                vEl.play().catch(() => {});
            }
        } else {
            var aEl = document.getElementById('audio');
            if (aEl) {
                aEl.srcObject = new MediaStream([evt.track]);
                // Start muted, then let the parent-selected playback mode
                // decide whether this clock-synchronised track is audible.
                aEl.muted = true;
                if (typeof window.xiaomanApplyPlaybackMode === 'function') {
                    window.xiaomanApplyPlaybackMode();
                } else {
                    aEl.play().catch(() => {});
                }
            }
        }
    });

    var startEl = document.getElementById('start');
    if (startEl) startEl.style.display = 'none';

    var promise = negotiate();

    var stopEl = document.getElementById('stop');
    if (stopEl) stopEl.style.display = 'inline-block';

    return promise;
}

function stop() {
    var stopEl = document.getElementById('stop');
    if (stopEl) stopEl.style.display = 'none';
    stopSessionHeartbeat();

    var sidEl = document.getElementById('sessionid');
    var sessionId = sidEl && sidEl.value ? String(sidEl.value).trim() : '';
    if (sessionId && sessionId !== '0') {
        var endpoint = ((window.location.origin && window.location.origin !== 'null')
            ? window.location.origin : 'http://localhost:8010') + '/api/session/close';
        var body = JSON.stringify({sessionid: sessionId});
        if (navigator.sendBeacon) {
            navigator.sendBeacon(endpoint, new Blob([body], {type: 'application/json'}));
        } else {
            fetch(endpoint, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: body,
                keepalive: true,
            }).catch(() => {});
        }
        sidEl.value = '0';
    }

    if (pc) {
        pc.close();
        pc = null;
    }
}

window.addEventListener('pagehide', stop);
window.addEventListener('beforeunload', stop);
