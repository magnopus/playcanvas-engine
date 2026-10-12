import {
    BindingTwoWay,
    BooleanInput,
    Label,
    LabelGroup,
    Panel,
    Progress,
    SelectInput,
    SliderInput
} from '@playcanvas/pcui/react';
import { useEffect } from 'react';

/**
 * @import { Observer } from '@playcanvas/observer'
 * @import { ReactElement } from 'react'
 */

/**
 * A labelled memory bar: fill = observer[pctPath] (0-100), caption = observer[textPath].
 * The Progress element is driven imperatively - it has no binding support.
 *
 * @param {object} props - The props.
 * @param {Observer} props.observer - The panel observer.
 * @param {string} props.label - Row label.
 * @param {string} props.pctPath - Observer path of the fill percentage (0-100).
 * @param {string} props.textPath - Observer path of the caption text.
 * @returns {ReactElement} The bar row.
 */
function PoolBar({ observer, label, pctPath, textPath }) {
    const cls = `poolbar-${pctPath.replace(/\./g, '-')}`;
    useEffect(() => {
        const apply = (v) => {
            const ui = document.querySelector(`.${cls}`)?.ui;
            if (ui) ui.value = v;
        };
        apply(observer.get(pctPath) ?? 0);
        const h = observer.on(`${pctPath}:set`, apply);
        return () => h.unbind();
    }, [observer, pctPath, cls]);
    return (
        <>
            <LabelGroup text={label}>
                <Progress class={cls} value={0} />
            </LabelGroup>
            <LabelGroup text=''>
                <Label
                    binding={new BindingTwoWay()}
                    link={{ observer, path: textPath }}
                    value={observer.get(textPath)}
                />
            </LabelGroup>
        </>
    );
}

/**
 * @param {{ observer: Observer }} props - The control panel props.
 * @returns {ReactElement} The control panel.
 */
export function Controls({ observer }) {
    return (
        <>
            <Panel headerText='Meshlet Streaming'>
                <LabelGroup text='Colour'>
                    <SelectInput
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.colorMode' }}
                        type='number'
                        options={[
                            { v: 0, t: 'Material' },
                            { v: 2, t: 'Meshlet' },
                            { v: 1, t: 'LOD tier' }
                        ]}
                    />
                </LabelGroup>
                <LabelGroup text='Bunnies'>
                    <SliderInput
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.gridSize' }}
                        min={0}
                        max={100}
                        step={1}
                        precision={0}
                    />
                </LabelGroup>
                <LabelGroup text='Error px'>
                    <SliderInput
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.threshold' }}
                        min={0.25}
                        max={32}
                        precision={2}
                    />
                </LabelGroup>
                <LabelGroup text='Occlusion'>
                    <BooleanInput
                        type='toggle'
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.occlusion' }}
                    />
                </LabelGroup>
                <LabelGroup text='Lights'>
                    <BooleanInput
                        type='toggle'
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.lights' }}
                    />
                </LabelGroup>
                <LabelGroup text='IBL'>
                    <BooleanInput
                        type='toggle'
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.ibl' }}
                    />
                </LabelGroup>
                <LabelGroup text='Fog'>
                    <SelectInput
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.fog' }}
                        type='string'
                        options={[
                            { v: 'none', t: 'None' },
                            { v: 'linear', t: 'Linear' }
                        ]}
                    />
                </LabelGroup>
                <LabelGroup text='Bunnies on'>
                    <BooleanInput
                        type='toggle'
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.bunnies' }}
                    />
                </LabelGroup>
                <LabelGroup text='Occluder wall'>
                    <BooleanInput
                        type='toggle'
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.occluder' }}
                    />
                </LabelGroup>
                <LabelGroup text='Glass slab'>
                    <BooleanInput
                        type='toggle'
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.slab' }}
                    />
                </LabelGroup>
                <LabelGroup text='Shadows'>
                    <BooleanInput
                        type='toggle'
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.shadows' }}
                    />
                </LabelGroup>
                <LabelGroup text='Cascades'>
                    <SliderInput
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.cascades' }}
                        min={1}
                        max={4}
                        step={1}
                        precision={0}
                    />
                </LabelGroup>
                <LabelGroup text='Shadow dist'>
                    <SliderInput
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.shadowDist' }}
                        min={0}
                        max={200}
                        step={5}
                        precision={0}
                    />
                </LabelGroup>
                <LabelGroup text='Tex MB'>
                    <SliderInput
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.texPoolMb' }}
                        min={8}
                        max={256}
                        step={8}
                        precision={0}
                    />
                </LabelGroup>
                <LabelGroup text='Geo MB'>
                    <SliderInput
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.geoPoolMb' }}
                        min={128}
                        max={1536}
                        step={64}
                        precision={0}
                    />
                </LabelGroup>
                <PoolBar
                    observer={observer}
                    label='Geo pool'
                    pctPath='data.geoPct'
                    textPath='data.geoBar'
                />
                <PoolBar
                    observer={observer}
                    label='Tex pool'
                    pctPath='data.texPct'
                    textPath='data.texBar'
                />
                <LabelGroup text='Drawn'>
                    <Label
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.stats' }}
                        value={observer.get('data.stats')}
                    />
                </LabelGroup>
                <LabelGroup text='Budget'>
                    <Label
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.budgetWarning' }}
                        value={observer.get('data.budgetWarning')}
                    />
                </LabelGroup>
                <LabelGroup text='Textures'>
                    <Label
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.texStats' }}
                        value={observer.get('data.texStats')}
                    />
                </LabelGroup>
            </Panel>
        </>
    );
}
