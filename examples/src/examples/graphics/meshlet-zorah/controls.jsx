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
 * @param {object} props - The props.
 * @param {Observer} props.observer - The observer.
 * @returns {ReactElement} The control panel.
 */
export function Controls({ observer }) {
    return (
        <Panel headerText='Zorah'>
            <LabelGroup text='View'>
                <SelectInput
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.teleport' }}
                    type='string'
                    options={[
                        { v: 'hero', t: 'Hero view' },
                        { v: 'Centre', t: 'Centre' },
                        { v: 'North-west', t: 'North-west' },
                        { v: 'North-east', t: 'North-east' },
                        { v: 'South-west', t: 'South-west' },
                        { v: 'South-east', t: 'South-east' }
                    ]}
                />
            </LabelGroup>
            <LabelGroup text='Sun'>
                <SliderInput
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.sunIntensity' }}
                    min={0}
                    max={6}
                    precision={2}
                />
            </LabelGroup>
            <LabelGroup text='Sky'>
                <SliderInput
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.skyLuminance' }}
                    min={0.05}
                    max={1.5}
                    precision={2}
                />
            </LabelGroup>
            <LabelGroup text='Shadows'>
                <BooleanInput
                    type='toggle'
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.shadows' }}
                />
            </LabelGroup>
            <LabelGroup text='Static shadows'>
                <BooleanInput
                    type='toggle'
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.staticShadows' }}
                />
            </LabelGroup>
            <LabelGroup text='SSAO'>
                <BooleanInput
                    type='toggle'
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.ssao' }}
                />
            </LabelGroup>
            <LabelGroup text='TAA'>
                <BooleanInput
                    type='toggle'
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.taa' }}
                />
            </LabelGroup>
            <LabelGroup text='Occlusion'>
                <BooleanInput
                    type='toggle'
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.occlusion' }}
                />
            </LabelGroup>
            <LabelGroup text='Occluded roots'>
                <BooleanInput
                    type='toggle'
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.occlusionAdmission' }}
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
            <LabelGroup text='In view'>
                <Label binding={new BindingTwoWay()} link={{ observer, path: 'data.stats' }} />
            </LabelGroup>
            <LabelGroup text='Roots'>
                <Label binding={new BindingTwoWay()} link={{ observer, path: 'data.statsRoots' }} />
            </LabelGroup>
            <LabelGroup text='Demand'>
                <Label binding={new BindingTwoWay()} link={{ observer, path: 'data.statsDemand' }} />
            </LabelGroup>
            <LabelGroup text='LOD pressure'>
                <Label binding={new BindingTwoWay()} link={{ observer, path: 'data.statsPressure' }} />
            </LabelGroup>
            <LabelGroup text='Pages'>
                <Label binding={new BindingTwoWay()} link={{ observer, path: 'data.statsPages' }} />
            </LabelGroup>
        </Panel>
    );
}
