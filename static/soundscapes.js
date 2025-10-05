// Adaptive soundscapes using WebAudio with gentle fade
export class Soundscape {
  constructor() {
    this.ctx = null; this.gain = null; this.audio = null; this.src = null;
  }
  async init() {
    if (this.ctx) return; this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    this.gain = this.ctx.createGain(); this.gain.gain.value = 0; this.gain.connect(this.ctx.destination);
    this.audio = new Audio(); this.audio.loop = true;
    this.src = this.ctx.createMediaElementSource(this.audio); this.src.connect(this.gain);
  }
  async play(url, volume=0.2) {
    await this.init();
    if (this.audio.src !== url) this.audio.src = url;
    await this.audio.play();
    this.fadeTo(volume, 600);
  }
  pause() { if (!this.ctx) return; this.fadeTo(0, 600); setTimeout(() => this.audio.pause(), 650); }
  fadeTo(target, ms) {
    if (!this.gain) return; const now = this.ctx.currentTime; this.gain.gain.cancelScheduledValues(now);
    this.gain.gain.setValueAtTime(this.gain.gain.value, now);
    this.gain.gain.linearRampToValueAtTime(target, now + ms/1000);
  }
}

