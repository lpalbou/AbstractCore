/**
 * Shared UI Components for AbstractCore
 * Ensures consistent branding across all pages
 */

/**
 * Generate the AbstractCore animated logo HTML
 * @param {string} linkHref - The href for the brand link (e.g., "../index.html" or "#")
 * @returns {string} HTML string for the animated logo
 */
function getAbstractCoreLogo(linkHref = "#") {
    return `
        <a href="${linkHref}" class="brand-link">
            <div class="brand-logo">
                <div class="logo-abstract">
                    <div class="logo-circle">
                        <div class="orbit-container">
                            <div class="orbit-dot orbit-dot-1"></div>
                            <div class="orbit-dot orbit-dot-2"></div>
                        </div>
                    </div>
                    <div class="logo-lines">
                        <div class="logo-line"></div>
                        <div class="logo-line"></div>
                        <div class="logo-line"></div>
                    </div>
                </div>
            </div>
            <span class="brand-text">AbstractCore</span>
        </a>
    `;
}

/**
 * Initialize the AbstractCore logo in the navigation
 * Call this function after the DOM is loaded
 */
function initAbstractCoreLogo() {
    const navBrand = document.querySelector('.nav-brand');
    if (navBrand) {
        // Determine the correct link href based on current page
        const isRootPage = window.location.pathname === '/' || window.location.pathname.endsWith('/index.html') || window.location.pathname.endsWith('/');
        const linkHref = isRootPage ? "#" : "../index.html";
        
        navBrand.innerHTML = getAbstractCoreLogo(linkHref);
    }
}

/**
 * Initialize all shared components
 */
function initSharedComponents() {
    initAbstractCoreLogo();
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSharedComponents);
} else {
    initSharedComponents();
}
