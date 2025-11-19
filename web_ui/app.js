document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('feature-form');
  const FIXED_THR = 0.5;
  const btnFill = document.getElementById('fill-example');
  const btnSubmit = document.getElementById('submit');
  const resultCard = document.getElementById('result-card');
  const elProb = document.getElementById('prob');
  const elThr = document.getElementById('thr');
  const elRisk = document.getElementById('risk');

  const FEATURE_NAMES = Array.isArray(window.FEATURE_NAMES) ? window.FEATURE_NAMES : [];

  // 固定阈值 0.5，无交互

  const SAMPLE = {
    // 常见正常参考值
    'CLCR': 100,         // mL/min（成年人常见范围约90-130）
    'height': 170,       // cm（示例）
    'Daily dose(g)': 0.6,// g（碳酸锂示例日剂量）
    'ALB(g/L)': 45,      // g/L（常见 40-50）
    'ALP(U/L)': 90,      // U/L（常见 44-147）
    'GGT(U/L)': 20,      // U/L（常见 ≤50 男，≤32 女）
    'Na(mmol/L)': 140,   // mmol/L（常见 136-145）
  };

  btnFill.addEventListener('click', (e) => {
    e.preventDefault();
    FEATURE_NAMES.forEach((name, idx) => {
      const input = document.getElementById(`f_${idx+1}`);
      if (!input) return;
      if (Object.prototype.hasOwnProperty.call(SAMPLE, name)) {
        input.value = SAMPLE[name];
      }
    });
  });

  btnSubmit.addEventListener('click', async (e) => {
    e.preventDefault();
    try {
      const inst = {};
      FEATURE_NAMES.forEach((name, idx) => {
        const input = document.getElementById(`f_${idx+1}`);
        if (!input) return;
        inst[name] = input.value;
      });

      const payload = { instances: [inst] };
      const threshold = FIXED_THR;
      const resp = await fetch(`/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await resp.json();
      if (!data.ok) {
        elProb.textContent = '—';
        elThr.textContent = thr.value;
        const ERR = (window.TEXTS && window.TEXTS.error) || '错误';
        elRisk.textContent = ERR;
        elRisk.classList.remove('ok');
        elRisk.classList.add('bad');
        resultCard.classList.remove('hidden');
        return;
      }

      const prob = Array.isArray(data.pos_proba) ? Number(data.pos_proba[0]) : NaN;
      const thrShow = Number(data.threshold ?? threshold);
      const pred = Array.isArray(data.pred) ? Number(data.pred[0]) : (prob >= thrShow ? 1 : 0);

      elProb.textContent = isNaN(prob) ? '—' : prob.toFixed(4);
      elThr.textContent = FIXED_THR.toFixed(2);
      const RISK_BAD = (window.TEXTS && window.TEXTS.risk_insufficient) || '低血药浓度风险';
      const RISK_OK  = (window.TEXTS && window.TEXTS.risk_adequate) || '血药浓度达标或偏高';
      elRisk.textContent = pred === 0 ? RISK_BAD : RISK_OK;
      elRisk.classList.remove('ok', 'bad');
      elRisk.classList.add(pred === 0 ? 'bad' : 'ok');

      resultCard.classList.remove('hidden');
      resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } catch (err) {
      elProb.textContent = '—';
      elThr.textContent = FIXED_THR.toFixed(2);
      elRisk.textContent = (window.TEXTS && window.TEXTS.error) || '错误';
      elRisk.classList.remove('ok');
      elRisk.classList.add('bad');
      resultCard.classList.remove('hidden');
      console.error('预测失败：', err);
    }
  });
});
