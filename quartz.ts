import { loadQuartzConfig, loadQuartzLayout } from "./quartz/plugins/loader/config-loader"
import HomepageSidebarImage from "./quartz/components/HomepageSidebarImage"
import ConditionalRender from "./quartz/components/ConditionalRender"
import { PageTypeDispatcher } from "./quartz/plugins/pageTypes/dispatcher"

const config = await loadQuartzConfig()
export default config

const layout = await loadQuartzLayout()
const sidebarImage = HomepageSidebarImage()

for (const pageLayout of [layout.defaults, ...Object.values(layout.byPageType)]) {
  const right = pageLayout.right
  if (right?.length) {
    const graphOutsideHomepage = ConditionalRender({
      component: right[0],
      condition: ({ fileData }) => fileData.slug !== "index",
    })
    pageLayout.right = [sidebarImage, graphOutsideHomepage, ...right.slice(1)]
  }
}

// Recreate the v5 page dispatcher so rendering uses the customized layout.
const dispatcherIndex = config.plugins.emitters.findIndex(
  (emitter) => emitter.name === "PageTypeDispatcher",
)
if (dispatcherIndex === -1) throw new Error("PageTypeDispatcher is missing")
config.plugins.emitters[dispatcherIndex] = PageTypeDispatcher(layout)

export { layout }
