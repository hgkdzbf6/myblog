所有页面的布局

layout
- partial/head
    - partial/post/analytics
- partial/navigation
- <%- body %>
- partial/footer
- partial/search
- partial/after_footer
    - fancybox已经停止服务了qwq

有一些东西，比如说page，theme啥的变量，不知道从什么地方来。

从body里面来的东西，比如

index
- page-header, page-header-inverse
- _partial/index(row page)
    - col-md-9
    - partial/post/slogan
    - top_search
    - mypage
    - 标题和条目
    - 首先渲染置顶的标签
        - partial/post/title_top
        - partial/post/entry
    - 然后渲染其他文章
        - partial/post/title
            - h1 title
            - 
        - partial/post/entry
    - partial/index_pagination(center)分页