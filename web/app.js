const references = [
  {
    title: "WAM 数学化学习笔记",
    pages: 3,
    file: "../note/source/wam_math_notes.pdf",
    cover: "assets/wam_math_notes-1.png"
  },
  {
    title: "World Model 与 WAM",
    pages: 3,
    file: "../note/source/wm_vs_wam_notes.pdf",
    cover: "assets/wm_vs_wam_notes-1.png"
  },
  {
    title: "Model / Planning / Rollout",
    pages: 4,
    file: "../note/source/model_planning_synthetic_rollout.pdf",
    cover: "assets/model_planning_synthetic_rollout-1.png"
  },
  {
    title: "WAM 三种典型范式",
    pages: 7,
    file: "../note/source/wam_three_paradigms.pdf",
    cover: "assets/wam_three_paradigms-1.png"
  },
  {
    title: "WAM 系统学习笔记",
    pages: 5,
    file: "../note/source/wam_study_notes.pdf",
    cover: "assets/wam_study_notes-1.png"
  }
];

const grid = document.querySelector("#referenceGrid");

grid.innerHTML = references.map((item) => `
  <article class="reference-card">
    <img src="${item.cover}" alt="${item.title} 封面">
    <div>
      <h3>${item.title}</h3>
      <p>${item.pages} pages · source note</p>
      <a href="${item.file}" target="_blank" rel="noreferrer">打开 PDF</a>
    </div>
  </article>
`).join("");
