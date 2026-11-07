import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { pathToRoot } from "../util/path"

function HomepageSidebarImage({ fileData }: QuartzComponentProps) {
  if (fileData.slug !== "index") return null

  return (
    <figure class="homepage-sidebar-image desktop-only">
      <img
        src={`${pathToRoot(fileData.slug)}/static/homepage-sidebar.png`}
        alt="Rick 在异星风景中的马桶旁"
        width={853}
        height={762}
        decoding="async"
      />
      <figcaption>
        <a href="mailto:lkc22@mails.tsinghua.edu.cn">lkc22@mails.tsinghua.edu.cn</a>
      </figcaption>
    </figure>
  )
}

HomepageSidebarImage.css = `
.homepage-sidebar-image {
  margin: 0;
  width: 100%;
}
.homepage-sidebar-image img {
  display: block;
  width: 100%;
  height: auto;
  box-sizing: border-box;
  border: 1px solid var(--lightgray);
  border-radius: 12px;
}
.homepage-sidebar-image figcaption {
  margin-top: 0.75rem;
  font-size: 0.9rem;
  line-height: 1.5;
  text-align: center;
  overflow-wrap: anywhere;
}
.homepage-sidebar-image figcaption a {
  font-weight: normal;
}
`

export default (() => HomepageSidebarImage) satisfies QuartzComponentConstructor
