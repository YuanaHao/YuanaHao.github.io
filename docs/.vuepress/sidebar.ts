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
          text: "实验笔记",
          icon: "fa6-solid:feather-pointed",
          prefix: "/_posts/experiment/",
          link: "/blog",
          collapsible: true,
          children: [
            {
              text: "CS149",
              icon: "fa6-solid:feather-pointed",
              prefix: "/_posts/experiment/CS149/",
              link: "/blog",
              collapsible: true,
              children: ["/blog/CS149_asst1.md"]
            }
          ]
        }
      ]
    },
  ],
  // 专题区（独立侧边栏）
  "/apps/topic/": "structure",
});
