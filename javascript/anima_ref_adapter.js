(function () {
    "use strict";

    function syncThemeClass() {
        const isLobe =
            document.documentElement.dataset.lobeForge === "true" ||
            Boolean(document.querySelector(".lobe-sidebar-panel, link[href*='lobe-theme']"));

        document.querySelectorAll(".anima-ref-root, .anima-ref-accordion").forEach((el) => {
            el.classList.toggle("anima-ui-lobe", isLobe);
            el.classList.toggle("anima-ui-classic", !isLobe);
        });
        document.querySelectorAll(".anima-ref-root :is(.anima-status-text, .anima-ref-info, .anima-action-feedback)").forEach((el) => {
            el.setAttribute("role", "status");
        });
    }

    if (typeof onUiLoaded === "function") {
        onUiLoaded(syncThemeClass);
    }
    document.addEventListener("DOMContentLoaded", syncThemeClass);
})();
