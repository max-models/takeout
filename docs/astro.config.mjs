// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/takeout/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

export default defineDocs({
	repo: 'takeout',
	title: 'takeout',
	description: 'The output format and field maps shared by the max-models codes.',
	pythonPackage: 'takeout',
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Examples', link: '/examples/' },
		{ label: 'Development', link: '/development/docs/' },
		{ label: 'API', link: '/api/takeout/' },
	],
	sidebar: [
		{
			label: 'Getting started',
			items: [
				{ label: 'Installation', slug: 'getting-started/installation' },
				{ label: 'Quickstart', slug: 'getting-started/quickstart' },
			],
		},
		{
			label: 'Examples',
			items: [
				{ autogenerate: { directory: 'examples' } },
				{ label: 'Scripts', items: [{ autogenerate: { directory: 'example-scripts' } }] },
			],
		},
		{
			label: 'Development',
			items: [
				{ label: 'Building the docs', slug: 'development/docs' },
				{ label: 'Publishing', slug: 'development/publishing' },
			],
		},
	],
});
