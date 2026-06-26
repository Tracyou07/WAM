const notes = [
  {
    title: "WAM 数学化学习笔记",
    pages: 3,
    file: "../note/source/wam_math_notes.pdf",
    cover: "assets/wam_math_notes-1.png",
    summary: "从 POMDP 出发定义观测、动作、语言条件和 WAM 的联合建模对象。"
  },
  {
    title: "World Model 与 WAM 的区别",
    pages: 3,
    file: "../note/source/wm_vs_wam_notes.pdf",
    cover: "assets/wm_vs_wam_notes-1.png",
    summary: "区分世界动力学建模、动作生成和面向控制的世界-动作联合建模。"
  },
  {
    title: "Model, Planning 与 Synthetic Rollout",
    pages: 4,
    file: "../note/source/model_planning_synthetic_rollout.pdf",
    cover: "assets/model_planning_synthetic_rollout-1.png",
    summary: "解释 learned model、虚拟轨迹、规划目标、价值估计和误差累积。"
  },
  {
    title: "World Action Models 的三种典型范式",
    pages: 7,
    file: "../note/source/wam_three_paradigms.pdf",
    cover: "assets/wam_three_paradigms-1.png",
    summary: "比较 Joint Video-Action、Imagine-Then-Execute 和 Fast-WAM。"
  },
  {
    title: "WAM 系统学习笔记",
    pages: 5,
    file: "../note/source/wam_study_notes.pdf",
    cover: "assets/wam_study_notes-1.png",
    summary: "把核心概念合并成总览式笔记，适合作为汇报前的快速复盘。"
  }
];

const grid = document.querySelector("#noteGrid");
const frame = document.querySelector("#pdfFrame");
const readerTitle = document.querySelector("#readerTitle");
const openPdf = document.querySelector("#openPdf");

function setReader(note) {
  frame.src = note.file;
  readerTitle.textContent = note.title;
  openPdf.href = note.file;
  document.querySelector("#reader").scrollIntoView({ behavior: "smooth" });
}

grid.innerHTML = notes.map((note, index) => `
  <article class="note-card">
    <img src="${note.cover}" alt="${note.title} 封面">
    <div class="note-content">
      <div class="note-meta">${String(index + 1).padStart(2, "0")} · ${note.pages} pages</div>
      <h3>${note.title}</h3>
      <p>${note.summary}</p>
      <div class="note-actions">
        <button type="button" data-index="${index}">预览</button>
        <a href="${note.file}" target="_blank" rel="noreferrer">打开</a>
      </div>
    </div>
  </article>
`).join("");

grid.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-index]");
  if (!button) return;
  setReader(notes[Number(button.dataset.index)]);
});
