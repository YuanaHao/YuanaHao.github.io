import { sidebar } from "vuepress-theme-hope";

// 图标：https://theme-hope.vuejs.press/zh/guide/interface/icon.html#%E8%AE%BE%E7%BD%AE%E5%9B%BE%E6%A0%87
// https://fontawesome.com/search?m=free&o=r
export default sidebar({
  "": [
    {
      text: "博客文章",
      icon: "fa6-solid:feather-pointed",
      prefix: "/_posts/",
      link: "/blog",
      collapsible: true,
      children: [
        {
          text: "算法学习",
          link: "/blog/algorithm/",
        },
        {
          text: "实验笔记",
          link: "/blog/experiment/",
        },
        {
          text: "计算机优质课程",
          link: "/blog/course/",
        },
      ]
    },
  ],
  // 专题区（独立侧边栏）
  "/apps/topic/": "structure",
});
