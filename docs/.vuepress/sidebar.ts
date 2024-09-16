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
          prefix: "/_posts/algorithm/",
          link: "/blog",
        },
        {
          text: "实验笔记",
          prefix: "/_posts/expriment/",
          link: "/blog",
          collapsible: true,
          children: [
            {
              text: "CS149",
              prefix: "/_posts/expriment/CS149/",
              link: "/blog",
              collapsible: true,
              children: ["CS149_asst1.md"]
            }
          ],
        },
        {
          text: "计算机优质课程",
          prefix: "/_posts/course/",
          link: "/blog",
          collapsible: true,
          children: [
            {
              text: "CS149",
              prefix: "/_posts/course/CS149/",
              link: "/blog",
              collapsible: true,
              children: ["CS149.md"]
            }
          ],
        },
      ]
    },
  ],
  // 专题区（独立侧边栏）
  "/apps/topic/": "structure",
});
