---
layout: post
title: "깃허브로 블로그 만들기"
date: 2021-02-10
categories: git
tags: git github blog
---

### 동기
- 배운 것, 앞으로 배워나갈 것들을 기록할 곳이 필요했다.
- 개인적으로 깃허브 마크다운이 친숙했다.
- 마음만 먹으면 마음대로 이것저것 갖다 붙일 수 있다.
- 코드를 깔끔하게 기록할 수 있다.

### 첫 포스팅을 하기까지...
나는 [*Centrarium 테마*](http://jekyllthemes.org/themes/centrarium/)를 사용해 블로그를 개설했다. 금방 끝날 줄 알았는데 어째서인지 로컬에서는 잘 동작하던 pagination 기능이 깃허브에 배포되면 제대로 동작 하지않았다... 알고 보니 이 블로그가 사용중인 [*jekyll-paginate-v2 plugin*](https://github.com/sverrirs/jekyll-paginate-v2)이 [*github에서 제공하는 dependency 목록*](https://pages.github.com/versions/)에 없었기 때문이었다. ㅠㅠ

그래서 결국 [*프로젝트*](https://github.com/kjb4494/blog-builder)를 따로 만들어 빌드 결과물을 [*서브모듈*](https://github.com/kjb4494/kjb4494.github.io)로 두는 방법을 택했다. 즉, 빌드를 깃허브에 맡기지않고 결과물을 바로 올려 로컬환경과의 괴리를 없앴다. 결과적으로 푸시를 두 번 해야하는 귀찮음이 생겼지만 나중에 Jenkins를 쓰든, Travis를 쓰든지 해서 자동화 시켜야겠다...

---

### 깃허브로 블로그 만드는 법
- [본인아이디].github.io 로 레파지토리를 생성한다.
- [*깃허브 테마*](http://jekyllthemes.org/) 중에서 마음에 드는 테마를 골라 다운로드하고 레파지토리로 모두 복사한다.
- _config.yml 파일을 수정한다.
    ```yml
    # 이 부분은 반드시 본인아이디에 맞게 고쳐준다.
    url: "https://[본인아이디].github.io"
    ```
- commit & push 하고 조금만 기다리면 https://[본인아이디].github.io에 접속할 수 있다.
    >* 만약 안 될 경우 깃허브 프로젝트 settings에서 다음과 같이 main branch로 바꿔준다.
    ![사진](/assets/imgs/posts/git/create-github-blog-001.png)

여기까지가 가장 간단하게 깃허브를 이용해 블로그를 생성할 수 있는 방법인듯하다.

### github dependency 에서 벗어나기
자신이 원하는 플러그인을 마음껏 쓰려면 빌드 결과물을 배포해야한다. github.io를 서브모듈 프로젝트로 두고 [*부모 프로젝트*](https://github.com/kjb4494/blog-builder)를 생성했다.
- _config.yml 에서 destination 을 설정해준다.
    ```yml
    destination: ./output
    ```
그럼 빌드 결과물이 output 폴더에 생성된다. 
- 빌드하기 전에 위 폴더명으로 서브모듈을 달아준다.
```shell
git submodule add https://github.com/[본인아이디]/[본인아이디].github.io.git output
```
서브모듈이 성공적으로 적용됐으면 빌드 후 서브모듈의 레파지토리 안에서 commit & push 해주면 로컬에서 테스트한 그대로 블로그에 반영된다.
```shell
bundle exec jekyll serve
```

### code block 꾸미기
- _config.yml 에서 highlighter 설정하기
    ```yml
    highlighter: rouge
    ```
- rouge 설치 및 highligter css 다운받기
    ```shell
    gem install rouge
    rougify style base16.dark > tmp.css
    ```
    위 명령어를 수행하면 tmp.css 파일이 생성된다.
    이 파일의 내용을 _sass 폴더에서 적용시키면 된다.
- 사용할 수 있는 테마 목록 출력
    ```shell
    rougify help style
    ```
- font 바꾸기  
[*d2coding*](https://github.com/Joungkyun/font-d2coding) 웹 폰트를 적용해보았다. 다운받아서 메뉴얼대로 적용하면 된다.

### 후기
여러가지 삽질하느라 힘들었다 ㅠㅠ... pagination이 안 되는 이유 찾는게 시작이었다. 뭔가를 시작할 땐 공식문서를 필독하는 습관을 길러야겠다... 처음엔 금방 끝날 줄 알았는데 어쩌다보니 하루종일 해버리고 말았다. 

어쨌든 이 글을 시작으로 올해는 뿌듯한 개발자 라이프를 가졌으면 좋겠다. :)