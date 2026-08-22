var pc = null;

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

    if (pc) {
        setTimeout(() => {
            pc.close();
        }, 500);
    }
}

window.onunload = function(event) {
    // 在这里执行你想要的操作
    setTimeout(() => {
        pc.close();
    }, 500);
};

window.onbeforeunload = function (e) {
        setTimeout(() => {
                pc.close();
            }, 500);
        e = e || window.event
        // 兼容IE8和Firefox 4之前的版本
        if (e) {
          e.returnValue = '关闭提示'
        }
        // Chrome, Safari, Firefox 4+, Opera 12+ , IE 9+
        return '关闭提示'
      }
