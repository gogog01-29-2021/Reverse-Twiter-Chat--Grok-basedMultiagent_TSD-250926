const mediaQueryPx = {
  mobile: '768px',
  tablet: '1024px',
  desktop: '1200px',
  landscape: '(orientation: landscape)',
  portrait: '(orientation: portrait)',
};

const mediaQuery = {
  mobile: `all and (min-width: ${mediaQueryPx.mobile})`,
  tablet: `all and (min-width: ${mediaQueryPx.tablet})`,
  desktop: `all and (min-width: ${mediaQueryPx.desktop})`,
  landscape: `all and ${mediaQueryPx.landscape}`,
  portrait: `all and ${mediaQueryPx.portrait}`,
};

export default mediaQuery;
