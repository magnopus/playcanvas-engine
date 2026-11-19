import {
    BindingTwoWay,
    BooleanInput,
    Label,
    LabelGroup,
    Panel,
    SelectInput,
    SliderInput
} from '@playcanvas/pcui/react';

/**
 * @import { Observer } from '@playcanvas/observer'
 * @import { ReactElement } from 'react'
 */

/**
 * @param {{ observer: Observer }} props - The control panel props.
 * @returns {ReactElement} The control panel.
 */
export function Controls({ observer }) {
    return (
        <>
            <Panel headerText='Meshlet Inspect'>
                <LabelGroup text='Colour'>
                    <SelectInput
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.colorMode' }}
                        type='number'
                        options={[
                            { v: 0, t: 'Material' },
                            { v: 2, t: 'Meshlet' }
                        ]}
                    />
                </LabelGroup>
                <LabelGroup text='AA'>
                    <SelectInput
                        type='string'
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.aaMode' }}
                        options={[
                            { v: 'none', t: 'None' },
                            { v: 'taa', t: 'TAA' },
                            { v: 'msaa', t: 'MSAA 4x' }
                        ]}
                    />
                </LabelGroup>
                <LabelGroup text='Meshlets'>
                    <BooleanInput
                        type='toggle'
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.meshlets' }}
                    />
                </LabelGroup>
                <LabelGroup text='Bunnies'>
                    <SliderInput
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.gridSize' }}
                        min={1}
                        max={1000}
                        step={1}
                        precision={0}
                    />
                </LabelGroup>
                <LabelGroup text='Geo budget MB'>
                    <SliderInput
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.geoBudget' }}
                        min={0}
                        max={8192}
                        step={64}
                        precision={0}
                    />
                </LabelGroup>
                <LabelGroup text='Index share'>
                    <SliderInput
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.indexShare' }}
                        min={0.05}
                        max={0.9}
                        step={0.05}
                        precision={2}
                    />
                </LabelGroup>
                <LabelGroup text='Budget'>
                    <Label
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.budgetStats' }}
                        value={observer.get('data.budgetStats')}
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
                <LabelGroup text='Sun'>
                    <BooleanInput
                        type='toggle'
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.sunLight' }}
                    />
                </LabelGroup>
                <LabelGroup text='Spot'>
                    <BooleanInput
                        type='toggle'
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.spotLight' }}
                    />
                </LabelGroup>
                <LabelGroup text='Omnis'>
                    <BooleanInput
                        type='toggle'
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.omniLights' }}
                    />
                </LabelGroup>
                <LabelGroup text='IBL'>
                    <BooleanInput
                        type='toggle'
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.ibl' }}
                    />
                </LabelGroup>
                <LabelGroup text='Shadows'>
                    <BooleanInput
                        type='toggle'
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.shadows' }}
                    />
                </LabelGroup>
                <LabelGroup text='2nd view'>
                    <BooleanInput
                        type='toggle'
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.secondView' }}
                    />
                </LabelGroup>
                <LabelGroup text='Selected'>
                    <Label
                        binding={new BindingTwoWay()}
                        link={{ observer, path: 'data.selection' }}
                        value={observer.get('data.selection')}
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
            </Panel>
        </>
    );
}
